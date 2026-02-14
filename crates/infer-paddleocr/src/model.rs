use std::{
    convert::TryFrom,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, anyhow, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::DynamicImage;
use tokenizers::Tokenizer;

use crate::{
    config::{LoadedPaddleConfig, PaddleOcrVlConfig, load_config},
    snapshot::{self, AdapterScope, SnapshotLinearMap, SnapshotLoadPlan},
    transformer::ErnieDecoder,
    vision::{SiglipPreprocessConfig, SiglipProjector, SiglipVisionModel, preprocess_image},
};
use deepseek_ocr_core::{
    PromptBuildOptions,
    inference::{
        DecodeOutcome, DecodeParameters, ModelKind, ModelLoadArgs, OcrEngine, VisionSettings,
        normalize_text,
    },
    build_prompt_tokens_with,
    grid_token_count,
    sampling::{init_rng, select_token_id},
    tensor::{
        concat_token_embeddings, gather_token_embeddings, inject_embeddings_by_mask,
        to_dtype_if_needed,
    },
};

pub const DEFAULT_WEIGHTS_PATH: &str = "PaddleOCR-VL/model.safetensors";
const FALLBACK_EOS_TOKEN: &str = "</s>";

pub struct PaddleOcrModel {
    config: Arc<PaddleOcrVlConfig>,
    config_path: PathBuf,
    device: Device,
    dtype: DType,
    weights_path: PathBuf,
    vision: SiglipVisionModel,
    projector: SiglipProjector,
    decoder: ErnieDecoder,
}

#[derive(Debug)]
struct ProjectedImage {
    embeddings: Tensor,
    original_grid: (usize, usize, usize),
    merged_grid: (usize, usize, usize),
}

impl ProjectedImage {
    fn token_count(&self) -> usize {
        let (t, h, w) = self.merged_grid;
        t * h * w
    }

    fn split_original_grid(&self) -> (usize, usize, usize) {
        self.original_grid
    }

    fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }
}

struct PreparedPrompt {
    embeddings: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    context_tokens: Vec<i64>,
    next_position_base: i64,
}

impl PreparedPrompt {
    fn prompt_len(&self) -> usize {
        self.context_tokens.len()
    }
}

impl PaddleOcrModel {
    pub fn load(args: &ModelLoadArgs<'_>) -> Result<Self> {
        let ModelLoadArgs {
            device,
            dtype,
            weights_path,
            snapshot_path,
            ..
        } = args;
        let LoadedPaddleConfig { value, path } = load_config(args.config_path)?;
        let config = Arc::new(value);
        let resolved_weights = weights_path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_WEIGHTS_PATH));
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[resolved_weights.as_path()], *dtype, device)
        }
        .with_context(|| format!("failed to mmap weights at {}", resolved_weights.display()))?;
        let (mut snapshot_hits, snapshot_label) =
            load_snapshot_hits(config.as_ref(), device, *snapshot_path)
                .context("failed to prepare snapshot hits")?;
        let vision = SiglipVisionModel::load(
            &vb,
            &config.vision_config,
            *dtype,
            snapshot_hits.as_mut(),
            snapshot_label,
        )
        .context("failed to load SigLIP vision model")?;
        let projector = SiglipProjector::load(
            &vb,
            &config.vision_config,
            config.hidden_size,
            *dtype,
            snapshot_hits.as_mut(),
            snapshot_label,
        )
        .context("failed to load projector module")?;
        let decoder = ErnieDecoder::load(
            Arc::clone(&config),
            &vb,
            snapshot_hits.as_mut(),
            snapshot_label,
        )
        .context("failed to load Ernie decoder")?;
        Ok(Self {
            config,
            config_path: path,
            device: device.clone(),
            dtype: *dtype,
            weights_path: resolved_weights,
            vision,
            projector,
            decoder,
        })
    }

    pub fn config(&self) -> &PaddleOcrVlConfig {
        self.config.as_ref()
    }

    pub fn config_path(&self) -> &Path {
        self.config_path.as_path()
    }

    pub fn projector(&self) -> &SiglipProjector {
        &self.projector
    }

    pub fn vision_model(&self) -> &SiglipVisionModel {
        &self.vision
    }

    pub fn decoder(&self) -> &ErnieDecoder {
        &self.decoder
    }

    #[allow(dead_code)]
    fn encode_image(
        &self,
        image: &DynamicImage,
        vision_settings: VisionSettings,
    ) -> Result<ProjectedImage> {
        let prep_cfg = SiglipPreprocessConfig::from_vision_config(&self.config.vision_config)
            .with_max_image_size(vision_settings.image_size);
        let patches = preprocess_image(image, &self.device, &prep_cfg)
            .context("failed to preprocess image for SigLIP")?;
        let vision_hidden = self
            .vision
            .forward(&patches, self.config.use_3d_rope, true, &self.device)
            .context("SigLIP encoder forward pass failed")?;
        let (batch, tokens, hidden) = vision_hidden
            .shape()
            .dims3()
            .context("vision encoder must return [batch, seq, hidden]")?;
        anyhow::ensure!(
            batch == 1,
            "SigLIP vision outputs expect batch size 1 per image (got {batch})"
        );
        let features = vision_hidden
            .reshape((tokens, hidden))?
            .contiguous()
            .context("vision features not contiguous")?;
        let projected = self
            .projector
            .project_single(&features, patches.grid_thw)
            .context("projector forward failed")?;
        Ok(ProjectedImage {
            embeddings: projected.embeddings,
            original_grid: patches.grid_thw,
            merged_grid: projected.grid,
        })
    }

    fn encode_images(
        &self,
        images: &[DynamicImage],
        vision_settings: VisionSettings,
    ) -> Result<Vec<ProjectedImage>> {
        images
            .iter()
            .enumerate()
            .map(|(idx, image)| {
                self.encode_image(image, vision_settings)
                    .with_context(|| format!("failed to encode image {idx}"))
            })
            .collect()
    }

    fn prepare_prompt(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        projected: &[ProjectedImage],
    ) -> Result<PreparedPrompt> {
        let grids: Vec<(usize, usize, usize)> = projected
            .iter()
            .map(ProjectedImage::split_original_grid)
            .collect();
        let (input_tokens, image_mask_vec) =
            build_prompt_tokens(tokenizer, prompt, &grids, &self.config)?;
        ensure!(
            !input_tokens.is_empty(),
            "prompt must produce at least one token"
        );
        let placeholder_count = image_mask_vec.iter().filter(|&&flag| flag != 0).count();
        let available_tokens: usize = projected.iter().map(ProjectedImage::token_count).sum();
        ensure!(
            placeholder_count == available_tokens,
            "image placeholder span ({placeholder_count}) mismatches projector outputs ({available_tokens})"
        );

        let prompt_len = input_tokens.len();
        let device = &self.device;
        let input_ids = Tensor::from_vec(input_tokens.clone(), (1, prompt_len), device)?
            .to_dtype(DType::I64)?;
        let attention_mask = Tensor::ones((1, prompt_len), DType::U8, device)?;
        let image_mask = Tensor::from_vec(image_mask_vec.clone(), (1, prompt_len), device)?
            .to_dtype(DType::U8)?;
        let (position_ids, deltas) =
            compute_position_ids(&self.config, &input_ids, Some(&attention_mask), &[grids])?;
        let delta_host = deltas.to_vec2::<i64>()?;
        ensure!(
            delta_host.len() == 1 && delta_host[0].len() == 1,
            "delta tensor must have shape [batch, 1]"
        );
        let next_position_base = prompt_len as i64 + delta_host[0][0];

        let base_embeddings = gather_token_embeddings(self.decoder.embed_tokens(), &input_ids)?;
        let replacements = match flatten_image_embeddings(projected)? {
            Some(tensor) => vec![tensor],
            None => Vec::new(),
        };
        let fused_embeddings = inject_embeddings_by_mask(&base_embeddings, &image_mask, &replacements)?;

        Ok(PreparedPrompt {
            embeddings: fused_embeddings,
            attention_mask,
            position_ids,
            context_tokens: input_tokens,
            next_position_base,
        })
    }
}

fn load_snapshot_hits(
    cfg: &PaddleOcrVlConfig,
    device: &Device,
    snapshot_path: Option<&Path>,
) -> Result<(Option<SnapshotLinearMap>, Option<&'static str>)> {
    let Some(path) = snapshot_path else {
        return Ok((None, None));
    };
    let snapshot = snapshot::QuantizedSnapshot::load(path)
        .with_context(|| format!("failed to load snapshot from {}", path.display()))?;
    let specs = snapshot::paddle_snapshot_specs(cfg, AdapterScope::TextAndProjector)
        .context("failed to derive Paddle snapshot specs")?;
    if specs.is_empty() {
        return Ok((None, Some(snapshot.container_label())));
    }
    let plan = SnapshotLoadPlan::new(specs);
    let hits = plan
        .execute(Some(&snapshot), device, None)?
        .unwrap_or_default();
    Ok((Some(hits), Some(snapshot.container_label())))
}

impl OcrEngine for PaddleOcrModel {
    fn kind(&self) -> ModelKind {
        ModelKind::PaddleOcrVl
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> candle_core::DType {
        self.dtype
    }

    fn weights_path(&self) -> Option<&Path> {
        Some(self.weights_path.as_path())
    }

    fn flash_attention_enabled(&self) -> bool {
        self.config.use_flash_attention
    }

    fn decode(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        images: &[DynamicImage],
        vision: VisionSettings,
        params: &DecodeParameters,
        stream: Option<&dyn Fn(usize, &[i64])>,
    ) -> Result<DecodeOutcome> {
        ensure!(
            params.use_cache,
            "PaddleOCR decoder currently requires use_cache=true"
        );
        let eos_token_id = resolve_eos_token_id(self.config(), tokenizer);
        let encoded_images = self.encode_images(images, vision)?;
        let prepared = self.prepare_prompt(tokenizer, prompt, &encoded_images)?;
        if params.max_new_tokens == 0 {
            return Ok(DecodeOutcome {
                text: String::new(),
                prompt_tokens: prepared.prompt_len(),
                response_tokens: 0,
                generated_tokens: Vec::new(),
            });
        }

        let PreparedPrompt {
            embeddings,
            attention_mask,
            position_ids,
            mut context_tokens,
            mut next_position_base,
        } = prepared;

        let mut cache = self.decoder.new_cache();
        let mut guard = self.decoder.prompt_guard(&mut cache);
        let prefill = self.decoder.forward(
            None,
            Some(&embeddings),
            Some(&attention_mask),
            Some(&position_ids),
            Some(guard.cache()),
            params.use_cache,
        )?;
        let prompt_len = context_tokens.len();
        let logits = prefill.logits.get(0)?.get(prompt_len.saturating_sub(1))?;

        let mut rng = init_rng(params.seed);
        let mut generated = Vec::with_capacity(params.max_new_tokens);
        let mut current = select_token_id(&logits, params, &context_tokens, &mut rng)?;
        if let Some(eos) = eos_token_id
            && current == eos
        {
            return Ok(DecodeOutcome {
                text: String::new(),
                prompt_tokens: prompt_len,
                response_tokens: 0,
                generated_tokens: Vec::new(),
            });
        }

        while generated.len() < params.max_new_tokens {
            context_tokens.push(current);
            generated.push(current);
            if let Some(callback) = stream {
                callback(generated.len(), &generated);
            }
            if let Some(eos) = eos_token_id
                && current == eos
            {
                break;
            }
            if generated.len() >= params.max_new_tokens {
                break;
            }

            let decode_ids = single_token_tensor(current, &self.device)?;
            let decode_embeddings =
                gather_token_embeddings(self.decoder.embed_tokens(), &decode_ids)?;
            let pos_tensor = single_position_tensor(next_position_base, &self.device)?;
            next_position_base += 1;
            let decode = self.decoder.forward(
                None,
                Some(&decode_embeddings),
                None,
                Some(&pos_tensor),
                Some(guard.cache()),
                params.use_cache,
            )?;
            let next_logits = decode.logits.get(0)?.get(0)?;
            current = select_token_id(&next_logits, params, &context_tokens, &mut rng)?;
        }

        let decoded = tokenizer
            .decode(
                &generated
                    .iter()
                    .filter_map(|&id| u32::try_from(id).ok())
                    .collect::<Vec<_>>(),
                true,
            )
            .unwrap_or_default();
        let text = normalize_text(&decoded);
        Ok(DecodeOutcome {
            text,
            prompt_tokens: prompt_len,
            response_tokens: generated.len(),
            generated_tokens: generated,
        })
    }
}

pub fn load_model(args: ModelLoadArgs<'_>) -> Result<Box<dyn OcrEngine>> {
    if args.kind != ModelKind::PaddleOcrVl {
        return Err(anyhow!("unsupported model kind: {:?}", args.kind));
    }
    let model = PaddleOcrModel::load(&args)?;
    Ok(Box::new(model))
}

pub fn projector_token_count(grid: (usize, usize, usize), merge_size: usize) -> Result<usize> {
    grid_token_count(grid, merge_size)
}

pub fn build_prompt_tokens(
    tokenizer: &Tokenizer,
    prompt: &str,
    grids: &[(usize, usize, usize)],
    cfg: &PaddleOcrVlConfig,
) -> Result<(Vec<i64>, Vec<u8>)> {
    let image_token_id = cfg
        .image_token_id
        .ok_or_else(|| anyhow!("config missing image_token_id"))?;
    let vision_start_id = cfg
        .vision_start_token_id
        .ok_or_else(|| anyhow!("config missing vision_start_token_id"))?;
    let merge = cfg.vision_config.spatial_merge_size;
    let vision_end_id = tokenizer.token_to_id("<|IMAGE_END|>").map(|id| id as i64);
    let prefix = cfg.bos_token_id.map_or_else(Vec::new, |bos| vec![bos]);

    let sequence = build_prompt_tokens_with(
        tokenizer,
        prompt,
        grids.len(),
        PromptBuildOptions::image_slots("grids").with_prefix(&prefix),
        |idx, tokens, mask| {
            let placeholders = projector_token_count(grids[idx], merge)?;
            tokens.push(vision_start_id);
            mask.push(0);
            tokens.extend(std::iter::repeat_n(image_token_id, placeholders));
            mask.extend(std::iter::repeat_n(1u8, placeholders));
            if let Some(end_id) = vision_end_id {
                tokens.push(end_id);
                mask.push(0);
            }
            Ok(())
        },
    )?;

    Ok((sequence.tokens, sequence.image_mask))
}

pub fn compute_position_ids(
    cfg: &PaddleOcrVlConfig,
    input_ids: &Tensor,
    attention_mask: Option<&Tensor>,
    image_grids: &[Vec<(usize, usize, usize)>],
) -> Result<(Tensor, Tensor)> {
    let (batch, seq_len) = input_ids.shape().dims2()?;
    ensure!(
        image_grids.len() == batch,
        "image grid metadata must track each batch row"
    );
    let ids = to_dtype_if_needed(input_ids, DType::I64)?;
    let ids_host = ids.to_vec2::<i64>()?;
    let mask_host = if let Some(mask) = attention_mask {
        ensure!(
            mask.shape().dims() == [batch, seq_len],
            "attention mask must match [batch, seq]"
        );
        let m = to_dtype_if_needed(mask, DType::U8)?;
        Some(m.to_vec2::<u8>()?)
    } else {
        None
    };

    let mut per_row_positions: Vec<Vec<[i64; 3]>> = Vec::with_capacity(batch);
    let mut deltas = Vec::with_capacity(batch);
    let image_token_id = cfg
        .image_token_id
        .ok_or_else(|| anyhow!("config missing image_token_id"))?;
    let has_images = image_grids.iter().any(|grids| !grids.is_empty());

    for (row_idx, ids_row) in ids_host.iter().enumerate() {
        let mask_vec = mask_host
            .as_ref()
            .map(|rows| rows[row_idx].clone())
            .unwrap_or_else(|| vec![1u8; seq_len]);
        ensure!(
            mask_vec.len() == seq_len,
            "mask length mismatch for batch row {row_idx}"
        );

        if has_images {
            let (positions, max_val) = build_mrope_positions_for_row(
                cfg,
                ids_row,
                &mask_vec,
                &image_grids[row_idx],
                image_token_id,
            )?;
            per_row_positions.push(positions);
            let delta = max_val + 1 - (ids_row.len() as i64);
            deltas.push(delta);
        } else if attention_mask.is_some() {
            let (positions, max_val) = build_masked_text_positions(ids_row.len(), &mask_vec)?;
            per_row_positions.push(positions);
            let delta = max_val + 1 - (seq_len as i64);
            deltas.push(delta);
        } else {
            let mut positions = Vec::with_capacity(seq_len);
            for idx in 0..seq_len {
                let val = idx as i64;
                positions.push([val, val, val]);
            }
            per_row_positions.push(positions);
            deltas.push(0);
        }
    }

    let mut axis_t = vec![1i64; batch * seq_len];
    let mut axis_h = axis_t.clone();
    let mut axis_w = axis_t.clone();
    for (batch_idx, positions) in per_row_positions.iter().enumerate() {
        ensure!(
            positions.len() == seq_len,
            "position vector length mismatch for row {batch_idx}"
        );
        let base = batch_idx * seq_len;
        for (idx, values) in positions.iter().enumerate() {
            axis_t[base + idx] = values[0];
            axis_h[base + idx] = values[1];
            axis_w[base + idx] = values[2];
        }
    }
    let device = input_ids.device();
    let time_tensor = Tensor::from_vec(axis_t, (batch, seq_len), device)?;
    let height_tensor = Tensor::from_vec(axis_h, (batch, seq_len), device)?;
    let width_tensor = Tensor::from_vec(axis_w, (batch, seq_len), device)?;
    let stacked = Tensor::stack(&[time_tensor, height_tensor, width_tensor], 0)?;
    let delta_tensor = Tensor::from_vec(deltas, (batch, 1), device)?;
    Ok((stacked, delta_tensor))
}

fn build_masked_text_positions(seq_len: usize, mask: &[u8]) -> Result<(Vec<[i64; 3]>, i64)> {
    let mut positions = Vec::with_capacity(seq_len);
    let mut current = 0i64;
    let mut max_val = 1i64;
    for &flag in mask {
        if flag != 0 {
            let val = current;
            positions.push([val, val, val]);
            max_val = max_val.max(val);
            current += 1;
        } else {
            positions.push([1, 1, 1]);
        }
    }
    Ok((positions, max_val))
}

fn build_mrope_positions_for_row(
    cfg: &PaddleOcrVlConfig,
    ids: &[i64],
    mask: &[u8],
    grids: &[(usize, usize, usize)],
    image_token_id: i64,
) -> Result<(Vec<[i64; 3]>, i64)> {
    let active_ids: Vec<i64> = ids
        .iter()
        .zip(mask.iter())
        .filter_map(|(&id, &flag)| (flag != 0).then_some(id))
        .collect();
    let mut axis_t = Vec::with_capacity(active_ids.len());
    let mut axis_h = Vec::with_capacity(active_ids.len());
    let mut axis_w = Vec::with_capacity(active_ids.len());
    let merge = cfg.vision_config.spatial_merge_size;
    let mut st = 0usize;
    let mut next_scalar = 0i64;
    let mut grid_iter = grids.iter();
    while st < active_ids.len() {
        let next_image = active_ids[st..].iter().position(|&id| id == image_token_id);
        match next_image {
            Some(offset) => {
                let ed = st + offset;
                append_text_chunk(&mut axis_t, &mut axis_h, &mut axis_w, next_scalar, ed - st);
                next_scalar += (ed - st) as i64;
                let grid = grid_iter
                    .next()
                    .ok_or_else(|| anyhow!("not enough image grids for placeholders"))?;
                let block = projector_token_count(*grid, merge)?;
                ensure!(
                    ed + block <= active_ids.len(),
                    "placeholder span exceeds token sequence"
                );
                ensure!(
                    active_ids[ed..ed + block]
                        .iter()
                        .all(|&id| id == image_token_id),
                    "non-image token encountered inside placeholder span"
                );
                append_vision_chunk(
                    cfg,
                    *grid,
                    merge,
                    next_scalar,
                    &mut axis_t,
                    &mut axis_h,
                    &mut axis_w,
                )?;
                next_scalar += block as i64;
                st = ed + block;
            }
            None => {
                append_text_chunk(
                    &mut axis_t,
                    &mut axis_h,
                    &mut axis_w,
                    next_scalar,
                    active_ids.len() - st,
                );
                next_scalar = active_ids.len() as i64;
                st = active_ids.len();
            }
        }
    }
    ensure!(
        grid_iter.next().is_none(),
        "unused image grids remain after placeholder expansion"
    );
    let max_val = axis_t
        .iter()
        .chain(axis_h.iter())
        .chain(axis_w.iter())
        .copied()
        .max()
        .unwrap_or(1);
    let mut positions = Vec::with_capacity(ids.len());
    let mut active_iter = axis_t
        .into_iter()
        .zip(axis_h)
        .zip(axis_w)
        .map(|((t, h), w)| [t, h, w]);
    for &flag in mask {
        if flag != 0 {
            let value = active_iter
                .next()
                .ok_or_else(|| anyhow!("insufficient active positions for mask entries"))?;
            positions.push(value);
        } else {
            positions.push([1, 1, 1]);
        }
    }
    Ok((positions, max_val))
}

fn append_text_chunk(
    axis_t: &mut Vec<i64>,
    axis_h: &mut Vec<i64>,
    axis_w: &mut Vec<i64>,
    base: i64,
    len: usize,
) {
    for offset in 0..len {
        let val = base + offset as i64;
        axis_t.push(val);
        axis_h.push(val);
        axis_w.push(val);
    }
}

fn flatten_image_embeddings(images: &[ProjectedImage]) -> Result<Option<Tensor>> {
    let owned: Vec<Tensor> = images
        .iter()
        .map(|image| image.embeddings().clone())
        .collect();
    concat_token_embeddings(owned)
}

pub fn resolve_eos_token_id(cfg: &PaddleOcrVlConfig, tokenizer: &Tokenizer) -> Option<i64> {
    cfg.eos_token_id.or_else(|| {
        tokenizer
            .token_to_id(FALLBACK_EOS_TOKEN)
            .map(|id| id as i64)
    })
}

fn single_token_tensor(token: i64, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(vec![token], (1, 1), device)?.to_dtype(DType::I64)?)
}

fn single_position_tensor(value: i64, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(vec![value, value, value], (3, 1, 1), device)?.to_dtype(DType::I64)?)
}

fn append_vision_chunk(
    cfg: &PaddleOcrVlConfig,
    grid: (usize, usize, usize),
    merge: usize,
    base: i64,
    axis_t: &mut Vec<i64>,
    axis_h: &mut Vec<i64>,
    axis_w: &mut Vec<i64>,
) -> Result<()> {
    let tokens_per_second = cfg.vision_config.tokens_per_second as f32;
    let (t, h, w) = grid;
    let llm_h = h / merge;
    let llm_w = w / merge;
    ensure!(
        llm_h * merge == h && llm_w * merge == w,
        "grid not divisible by merge size"
    );
    for temporal in 0..t {
        let time_val = ((temporal as f32) * 0.0 * tokens_per_second).floor() as i64;
        for row in 0..llm_h {
            for col in 0..llm_w {
                axis_t.push(base + time_val);
                axis_h.push(base + row as i64);
                axis_w.push(base + col as i64);
            }
        }
    }
    Ok(())
}
