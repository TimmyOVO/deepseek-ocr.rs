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
    inference::{
        DecodeOutcome, DecodeParameters, ModelKind, ModelLoadArgs, OcrEngine, VisionSettings,
        normalize_text,
    },
    sampling::{init_rng, select_token_id},
    tensor::gather_token_embeddings,
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
        let fused_embeddings =
            inject_image_embeddings(&base_embeddings, &image_mask, &replacements)?;

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
        if let Some(eos) = eos_token_id {
            if current == eos {
                return Ok(DecodeOutcome {
                    text: String::new(),
                    prompt_tokens: prompt_len,
                    response_tokens: 0,
                    generated_tokens: Vec::new(),
                });
            }
        }

        while generated.len() < params.max_new_tokens {
            context_tokens.push(current);
            generated.push(current);
            if let Some(callback) = stream {
                callback(generated.len(), &generated);
            }
            if let Some(eos) = eos_token_id {
                if current == eos {
                    break;
                }
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

#[allow(dead_code)]
pub(crate) fn projector_token_count(
    grid: (usize, usize, usize),
    merge_size: usize,
) -> Result<usize> {
    ensure!(merge_size > 0, "merge size must be positive");
    let (t, h, w) = grid;
    ensure!(
        h % merge_size == 0 && w % merge_size == 0,
        "grid {:?} not divisible by merge size {}",
        grid,
        merge_size
    );
    Ok(t * (h / merge_size) * (w / merge_size))
}

#[allow(dead_code)]
pub(crate) fn build_prompt_tokens(
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
    let segments: Vec<&str> = prompt.split("<image>").collect();
    ensure!(
        segments.len().saturating_sub(1) == grids.len(),
        "prompt/image mismatch: {} slots vs {} grids",
        segments.len().saturating_sub(1),
        grids.len()
    );

    let mut tokens = Vec::new();
    let mut mask = Vec::new();
    if let Some(bos) = cfg.bos_token_id {
        tokens.push(bos);
        mask.push(0);
    }

    for (idx, segment) in segments.iter().enumerate() {
        if !segment.is_empty() {
            let encoding = tokenizer
                .encode(*segment, false)
                .map_err(|err| anyhow!("tokenization failed: {err}"))?;
            tokens.extend(encoding.get_ids().iter().map(|&id| id as i64));
            mask.extend(std::iter::repeat(0u8).take(encoding.len()));
        }

        if idx < grids.len() {
            let placeholders = projector_token_count(grids[idx], merge)?;
            tokens.push(vision_start_id);
            mask.push(0);
            tokens.extend(std::iter::repeat(image_token_id).take(placeholders));
            mask.extend(std::iter::repeat(1u8).take(placeholders));
            if let Some(end_id) = vision_end_id {
                tokens.push(end_id);
                mask.push(0);
            }
        }
    }

    Ok((tokens, mask))
}

#[allow(dead_code)]
pub(crate) fn inject_image_embeddings(
    embeddings: &Tensor,
    mask: &Tensor,
    per_batch: &[Tensor],
) -> Result<Tensor> {
    let (batch, seq_len, hidden) = embeddings
        .shape()
        .dims3()
        .context("embeddings must have shape [batch, seq, hidden]")?;
    let mask = if mask.dtype() == DType::U8 {
        mask.clone()
    } else {
        mask.to_dtype(DType::U8)?
    };
    ensure!(
        mask.shape().dims() == [batch, seq_len],
        "image mask must have shape [batch, seq]"
    );

    let mut rows = Vec::with_capacity(batch);
    for b in 0..batch {
        let row = embeddings
            .get(b)?
            .reshape((seq_len, hidden))?
            .contiguous()?;
        let mask_row = mask.get(b)?.reshape((seq_len,))?;
        let mask_vec = mask_row.to_vec1::<u8>()?;
        let ones = mask_vec.iter().filter(|&&flag| flag != 0).count();
        if ones == 0 {
            rows.push(row);
            continue;
        }
        let replacements = per_batch
            .get(b)
            .ok_or_else(|| anyhow!("missing image embeddings for batch {b}"))?
            .to_dtype(row.dtype())?
            .to_device(row.device())?;
        let (rep_tokens, _) = replacements
            .shape()
            .dims2()
            .context("image embeddings must have shape [token_count, hidden_size]")?;
        ensure!(
            rep_tokens == ones,
            "image embeddings provide {rep_tokens} tokens but mask requires {ones}"
        );

        let mut rep_offset = 0usize;
        let mut cursor = 0usize;
        let mut segments = Vec::new();
        while cursor < seq_len {
            let flag = mask_vec[cursor];
            let start = cursor;
            while cursor < seq_len && mask_vec[cursor] == flag {
                cursor += 1;
            }
            let length = cursor - start;
            let segment = if flag == 0 {
                row.narrow(0, start, length)?
            } else {
                let seg = replacements.narrow(0, rep_offset, length)?;
                rep_offset += length;
                seg
            };
            segments.push(segment);
        }
        ensure!(
            rep_offset == ones,
            "not all replacement tokens were consumed (used {rep_offset} of {ones})"
        );
        let refs: Vec<&Tensor> = segments.iter().collect();
        rows.push(Tensor::cat(&refs, 0)?);
    }
    let refs: Vec<&Tensor> = rows.iter().collect();
    Ok(Tensor::stack(&refs, 0)?)
}

#[allow(dead_code)]
pub(crate) fn compute_position_ids(
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
    let ids = if input_ids.dtype() == DType::I64 {
        input_ids.clone()
    } else {
        input_ids.to_dtype(DType::I64)?
    };
    let ids_host = ids.to_vec2::<i64>()?;
    let mask_host = if let Some(mask) = attention_mask {
        ensure!(
            mask.shape().dims() == [batch, seq_len],
            "attention mask must match [batch, seq]"
        );
        let m = if mask.dtype() == DType::U8 {
            mask.clone()
        } else {
            mask.to_dtype(DType::U8)?
        };
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
        } else if let Some(_) = attention_mask {
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
        .zip(axis_h.into_iter())
        .zip(axis_w.into_iter())
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
    if images.is_empty() {
        return Ok(None);
    }
    let mut owned: Vec<Tensor> = images
        .iter()
        .map(|image| image.embeddings().clone())
        .collect();
    if owned.is_empty() {
        return Ok(None);
    }
    let merged = if owned.len() == 1 {
        owned.pop().expect("len checked above")
    } else {
        let refs: Vec<&Tensor> = owned.iter().collect();
        Tensor::cat(&refs, 0)?
    };
    Ok(Some(merged))
}

fn resolve_eos_token_id(cfg: &PaddleOcrVlConfig, tokenizer: &Tokenizer) -> Option<i64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_config;
    use ahash::AHashMap;
    use candle_core::{DType, Device, Tensor};
    use ndarray::{Array2, Array3, Array5, Axis, s};
    use ndarray_npy::NpzReader;
    use serde::Deserialize;
    use std::{
        fs::File,
        path::{Path, PathBuf},
    };
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    fn asset_path(relative: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate has parent")
            .parent()
            .expect("workspace root")
            .join(relative)
    }

    const SAMPLE_DOC_IMAGES: [&str; 1] = ["baselines/fixtures/paddleocr_vl/fixture_image.png"];
    const MULTI_IMAGE_DOC_IMAGES: [&str; 2] = [
        "baselines/fixtures/paddleocr_vl/fixture_image.png",
        "baselines/fixtures/paddleocr_vl/fixture_receipt.png",
    ];
    const LONG_PROMPT_IMAGES: [&str; 1] = ["baselines/fixtures/paddleocr_vl/fixture_image.png"];

    struct FixtureCase {
        name: &'static str,
        npz: &'static str,
        images: &'static [&'static str],
    }

    const PADDLE_FIXTURES: &[FixtureCase] = &[
        FixtureCase {
            name: "sample_doc",
            npz: "baselines/fixtures/paddleocr_vl/sample_doc.npz",
            images: &SAMPLE_DOC_IMAGES,
        },
        FixtureCase {
            name: "multi_image_doc",
            npz: "baselines/fixtures/paddleocr_vl/multi_image_doc.npz",
            images: &MULTI_IMAGE_DOC_IMAGES,
        },
        FixtureCase {
            name: "long_prompt_doc",
            npz: "baselines/fixtures/paddleocr_vl/long_prompt_doc.npz",
            images: &LONG_PROMPT_IMAGES,
        },
    ];

    fn build_test_tokenizer() -> Tokenizer {
        let mut vocab = AHashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("<|IMAGE_START|>".to_string(), 1);
        vocab.insert("<|IMAGE_END|>".to_string(), 2);
        vocab.insert("Question:".to_string(), 3);
        vocab.insert("Describe.".to_string(), 4);
        vocab.insert("User:".to_string(), 5);
        vocab.insert("end.".to_string(), 6);
        vocab.insert("</s>".to_string(), 7);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .expect("wordlevel model");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
        tokenizer
    }

    #[test]
    fn prompt_builder_matches_placeholder_tokens() -> Result<()> {
        let tokenizer = build_test_tokenizer();
        let loaded = load_config(Some(&asset_path("PaddleOCR-VL/config.json")))?;
        let cfg = loaded.value;
        let grid = (1usize, 16usize, 16usize);
        let (tokens, mask) =
            build_prompt_tokens(&tokenizer, "Question: <image> Describe.", &[grid], &cfg)?;
        let placeholders = mask.iter().filter(|&&flag| flag != 0).count();
        assert_eq!(
            placeholders,
            projector_token_count(grid, cfg.vision_config.spatial_merge_size)?
        );
        assert_eq!(tokens.len(), mask.len());
        Ok(())
    }

    #[test]
    fn position_ids_cover_image_span() -> Result<()> {
        let tokenizer = build_test_tokenizer();
        let loaded = load_config(Some(&asset_path("PaddleOCR-VL/config.json")))?;
        let cfg = loaded.value;
        let grid = (1usize, 16usize, 16usize);
        let (tokens, mask) = build_prompt_tokens(&tokenizer, "User: <image> end.", &[grid], &cfg)?;
        let seq_len = tokens.len();
        let ids =
            Tensor::from_vec(tokens.clone(), (1, seq_len), &Device::Cpu)?.to_dtype(DType::I64)?;
        let mask_tensor = Tensor::from_vec(mask.clone(), (1, seq_len), &Device::Cpu)?;
        let (position_ids, deltas) =
            compute_position_ids(&cfg, &ids, Some(&mask_tensor), &[vec![grid]])?;
        assert_eq!(position_ids.shape().dims(), [3, 1, seq_len]);
        assert_eq!(deltas.shape().dims(), [1, 1]);
        let time_axis = position_ids.get(0)?.get(0)?.to_vec1::<i64>()?;
        let image_token_id = cfg.image_token_id.unwrap();
        let first_image_index = tokens
            .iter()
            .position(|&id| id == image_token_id)
            .expect("image token present");
        assert_eq!(
            time_axis[first_image_index],
            time_axis[first_image_index + 1]
        );
        Ok(())
    }

    #[test]
    fn injection_replaces_masked_rows() -> Result<()> {
        let embeddings = Tensor::from_vec(
            (0..12).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 4, 3),
            &Device::Cpu,
        )?;
        let mask = Tensor::from_vec(vec![0u8, 1, 1, 0], (1, 4), &Device::Cpu)?;
        let replacements = Tensor::from_vec(
            vec![100f32, 101., 102., 200., 201., 202.],
            (2, 3),
            &Device::Cpu,
        )?;
        let result = inject_image_embeddings(&embeddings, &mask, &[replacements])?;
        let rows = result.to_vec3::<f32>()?;
        assert_eq!(rows[0][1][0], 100.0);
        assert_eq!(rows[0][2][2], 202.0);
        Ok(())
    }

    #[derive(Deserialize)]
    struct FixtureMetadata {
        prompt: String,
        images: Option<Vec<String>>,
    }

    #[test]
    fn tokenizer_fallback_supplies_eos_token() -> Result<()> {
        let tokenizer = build_test_tokenizer();
        assert_eq!(tokenizer.token_to_id("</s>"), Some(7));
        let mut loaded = load_config(Some(&asset_path("PaddleOCR-VL/config.json")))?;
        loaded.value.eos_token_id = None;
        let eos = super::resolve_eos_token_id(&loaded.value, &tokenizer);
        assert_eq!(eos, Some(7));
        Ok(())
    }

    #[test]
    fn paddle_fixture_matches_python_reference() -> Result<()> {
        let weights_path = asset_path("PaddleOCR-VL/model.safetensors");
        if !weights_path.exists() {
            eprintln!(
                "skipping PaddleOCR parity test: weights missing at {:?}",
                weights_path
            );
            return Ok(());
        }
        let tokenizer_path = asset_path("PaddleOCR-VL/tokenizer.json");
        if !tokenizer_path.exists() {
            eprintln!(
                "skipping PaddleOCR parity test: tokenizer missing at {:?}",
                tokenizer_path
            );
            return Ok(());
        }

        let device = Device::Cpu;
        let config_path = asset_path("PaddleOCR-VL/config.json");
        let args = ModelLoadArgs {
            kind: ModelKind::PaddleOcrVl,
            config_path: Some(config_path.as_path()),
            weights_path: Some(weights_path.as_path()),
            snapshot_path: None,
            device: device.clone(),
            dtype: DType::F32,
        };
        let model = PaddleOcrModel::load(&args)?;
        let prep_cfg = SiglipPreprocessConfig::from_vision_config(&model.config().vision_config);

        let tokenizer = match Tokenizer::from_file(tokenizer_path.to_string_lossy().as_ref()) {
            Ok(tok) => tok,
            Err(err) => {
                eprintln!("skipping PaddleOCR parity test: tokenizer load failed ({err})");
                return Ok(());
            }
        };

        let mut executed = 0usize;
        for case in PADDLE_FIXTURES {
            let fixture_npz = asset_path(case.npz);
            if !fixture_npz.exists() {
                eprintln!(
                    "skipping fixture {}: npz missing at {:?}",
                    case.name, fixture_npz
                );
                continue;
            }
            if let Some(missing_image) = case
                .images
                .iter()
                .map(|relative| asset_path(relative))
                .find(|path| !path.exists())
            {
                eprintln!(
                    "skipping fixture {}: image missing at {:?}",
                    case.name, missing_image
                );
                continue;
            }
            run_fixture_case(case, &fixture_npz, &model, &tokenizer, &prep_cfg)?;
            executed += 1;
        }

        if executed == 0 {
            eprintln!("skipping PaddleOCR parity test: no fixtures present");
        }
        Ok(())
    }

    fn run_fixture_case(
        case: &FixtureCase,
        fixture_npz: &Path,
        model: &PaddleOcrModel,
        tokenizer: &Tokenizer,
        prep_cfg: &SiglipPreprocessConfig,
    ) -> Result<()> {
        let fixture_file = File::open(fixture_npz).context("failed to open fixture npz")?;
        let mut reader = NpzReader::new(fixture_file).context("failed to parse fixture npz")?;
        let input_ids_np: Array2<i64> = reader
            .by_name("input_ids")
            .context("fixture missing input_ids")?;
        let attention_mask_np: Array2<i64> = reader
            .by_name("attention_mask")
            .context("fixture missing attention_mask")?;
        let position_ids_np: Array3<i64> = reader
            .by_name("position_ids")
            .context("fixture missing position_ids")?;
        let rope_deltas_np: Array2<i64> = reader
            .by_name("rope_deltas")
            .context("fixture missing rope_deltas")?;
        let image_grid_np: Array2<i64> = reader
            .by_name("image_grid_thw")
            .context("fixture missing image_grid_thw")?;
        let siglip_hidden_np: Array2<f32> = reader
            .by_name("siglip_hidden")
            .context("fixture missing siglip_hidden")?;
        let siglip_hidden_states_np: Option<Array3<f32>> =
            reader.by_name("siglip_hidden_states").ok();
        if siglip_hidden_states_np.is_some() {
            eprintln!("fixture {} includes siglip_hidden_states", case.name);
        }
        let pixel_values_np: Option<Array5<f32>> = reader.by_name("pixel_values_for_encoder").ok();
        let projector_np: Array2<f32> = reader
            .by_name("projector_embeddings")
            .context("fixture missing projector_embeddings")?;
        let fused_np: Array3<f32> = reader
            .by_name("fused_embeddings")
            .context("fixture missing fused_embeddings")?;
        let logits_np: Array2<f32> = reader
            .by_name("next_token_logits")
            .context("fixture missing next_token_logits")?;
        drop(reader);

        let fixture_json = fixture_npz.with_extension("json");
        let metadata: FixtureMetadata = serde_json::from_reader(
            File::open(&fixture_json).context("failed to read fixture metadata")?,
        )
        .context("failed to parse fixture metadata")?;
        let allow_state_drift = std::env::var_os("SIGLIP_ALLOW_STATE_DRIFT").is_some();

        let mut fixture_grids = Vec::new();
        for row in image_grid_np.axis_iter(Axis(0)) {
            ensure!(row.len() == 3, "grid row must have 3 entries");
            fixture_grids.push((row[0] as usize, row[1] as usize, row[2] as usize));
        }
        ensure!(
            fixture_grids.len() == case.images.len(),
            "fixture {} grid count ({}) mismatches image list ({})",
            case.name,
            fixture_grids.len(),
            case.images.len()
        );
        if let Some(images) = metadata.images {
            ensure!(
                images.len() == case.images.len(),
                "metadata lists {} images but fixture {} expects {}",
                images.len(),
                case.name,
                case.images.len()
            );
        }

        let mut patches = Vec::new();
        let mut pixel_offset = 0usize;
        for (idx, image_rel) in case.images.iter().enumerate() {
            let image_path = asset_path(image_rel);
            let image = image::open(&image_path).with_context(|| {
                format!("failed to open image {:?} for {}", image_path, case.name)
            })?;
            let processed = preprocess_image(&image, model.device(), prep_cfg)
                .with_context(|| format!("failed to preprocess {}", case.name))?;
            ensure!(
                processed.grid_thw == fixture_grids[idx],
                "preprocess grid {:?} mismatches fixture {:?} for {} (image {idx})",
                processed.grid_thw,
                fixture_grids[idx],
                case.name
            );
            if let Some(pixel_np) = pixel_values_np.as_ref() {
                let token_count =
                    processed.grid_thw.0 * processed.grid_thw.1 * processed.grid_thw.2;
                let end = pixel_offset + token_count;
                ensure!(
                    end <= pixel_np.shape()[1],
                    "pixel_values_for_encoder slice exceeds fixture bounds"
                );
                let slice = pixel_np.slice(s![0, pixel_offset..end, .., .., ..]);
                let expected: Vec<f32> = slice.iter().copied().collect();
                let (_, channels, patch_h, patch_w) = processed.patches.shape().dims4()?;
                let per_token = channels * patch_h * patch_w;
                let actual = processed
                    .patches
                    .to_dtype(DType::F32)?
                    .reshape((token_count * per_token,))?
                    .to_vec1::<f32>()?;
                ensure!(
                    actual.len() == expected.len(),
                    "pixel patch length mismatch ({} vs {})",
                    actual.len(),
                    expected.len()
                );
                let mut max_diff = 0f32;
                for (a, b) in actual.iter().zip(expected.iter()) {
                    max_diff = max_diff.max((a - b).abs());
                }
                eprintln!(
                    "{} pixel patch diff for image {} max abs {}",
                    case.name, idx, max_diff
                );
                ensure!(
                    max_diff <= 5e-3,
                    "{} pixel patch mismatch exceeds tolerance (image {idx})",
                    case.name
                );
                pixel_offset = end;
            }
            patches.push(processed);
        }

        let mut siglip_chunks = Vec::new();
        let mut projector_chunks = Vec::new();
        let mut per_layer_state_chunks: Option<Vec<Vec<Tensor>>> = None;
        let capture_states = siglip_hidden_states_np.is_some();
        let merge = model.config().vision_config.spatial_merge_size;
        for (patch, expected_grid) in patches.iter().zip(fixture_grids.iter()) {
            let (vision_hidden, state_list) = if capture_states {
                let (hidden, states) = model.vision_model().forward_with_states(
                    patch,
                    model.config().use_3d_rope,
                    true,
                    model.device(),
                )?;
                (hidden, Some(states))
            } else {
                (
                    model.vision_model().forward(
                        patch,
                        model.config().use_3d_rope,
                        true,
                        model.device(),
                    )?,
                    None,
                )
            };
            let (batch, tokens, hidden) = vision_hidden.shape().dims3()?;
            ensure!(batch == 1, "vision batch must be 1");
            let siglip_flat = vision_hidden
                .reshape((tokens, hidden))?
                .to_dtype(DType::F32)?
                .contiguous()?;
            siglip_chunks.push(siglip_flat.clone());

            if let Some(states) = state_list {
                if per_layer_state_chunks.is_none() {
                    per_layer_state_chunks = Some(vec![Vec::new(); states.len()]);
                }
                let buckets = per_layer_state_chunks
                    .as_mut()
                    .expect("state bucket initialized");
                ensure!(
                    buckets.len() == states.len(),
                    "state bucket count mismatch (expected {}, got {})",
                    buckets.len(),
                    states.len()
                );
                for (bucket, state) in buckets.iter_mut().zip(states.into_iter()) {
                    bucket.push(state);
                }
            }

            let projected = model
                .projector()
                .project_single(&siglip_flat, patch.grid_thw)?;
            ensure!(
                expected_grid.1 % merge == 0 && expected_grid.2 % merge == 0,
                "fixture grid {:?} not divisible by merge size {}",
                expected_grid,
                merge
            );
            let expected_projector_grid = (
                expected_grid.0,
                expected_grid.1 / merge,
                expected_grid.2 / merge,
            );
            ensure!(
                projected.grid == expected_projector_grid,
                "projector grid {:?} mismatches fixture {:?} for {}",
                projected.grid,
                expected_projector_grid,
                case.name
            );
            projector_chunks.push(projected.tokens().to_dtype(DType::F32)?.contiguous()?);
        }

        let siglip_tensor = concat_tensors(siglip_chunks)?.contiguous()?;
        if let (Some(state_chunks), Some(expected_states)) =
            (per_layer_state_chunks, siglip_hidden_states_np.as_ref())
        {
            let tol = 5e-3;
            ensure!(
                state_chunks.len() == expected_states.shape()[0],
                "fixture {} state count mismatch: expected {}, got {}",
                case.name,
                expected_states.shape()[0],
                state_chunks.len()
            );
            for (idx, (bucket, expected_slice)) in state_chunks
                .into_iter()
                .zip(expected_states.axis_iter(Axis(0)))
                .enumerate()
            {
                let tensor = concat_tensors(bucket)?.to_dtype(DType::F32)?.contiguous()?;
                let elem = tensor.shape().elem_count();
                let actual = tensor.reshape((elem,))?.to_vec1::<f32>()?;
                let expected = expected_slice
                    .as_slice()
                    .expect("siglip hidden state slice contiguous");
                ensure!(
                    actual.len() == expected.len(),
                    "{} siglip_hidden_states[{idx}] length mismatch ({} vs {})",
                    case.name,
                    actual.len(),
                    expected.len()
                );
                let mut max_diff = 0f32;
                let mut max_idx = 0usize;
                let mut max_vals = (0f32, 0f32);
                for (pos, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
                    let diff = (a - b).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        max_idx = pos;
                        max_vals = (*a, *b);
                    }
                }
                let hidden = tensor.shape().dims2()?.1;
                let token_idx = max_idx / hidden;
                let feature_idx = max_idx % hidden;
                eprintln!(
                    "{} siglip_hidden_states[{idx}] max abs diff {max_diff} (token {}, feature {}) rust={} python={}",
                    case.name, token_idx, feature_idx, max_vals.0, max_vals.1
                );
                if allow_state_drift {
                    eprintln!(
                        "{} siglip_hidden_states[{idx}] exceeds tol {} (max diff {})",
                        case.name, tol, max_diff
                    );
                } else {
                    ensure!(
                        max_diff <= tol,
                        "{} siglip_hidden_states[{idx}] max abs diff {} exceeds tolerance {}",
                        case.name,
                        max_diff,
                        tol
                    );
                }
            }
        }

        if let Some(expected_states) = siglip_hidden_states_np.as_ref() {
            let hidden_dim = expected_states.shape()[2];
            let mut offset = 0usize;
            for (img_idx, patch) in patches.iter().enumerate() {
                let token_count = patch.grid_thw.0 * patch.grid_thw.1 * patch.grid_thw.2;
                let end = offset + token_count;
                let embedding_slice = expected_states.slice(s![0, offset..end, ..]);
                let pyro_embed: Vec<f32> = embedding_slice.iter().copied().collect();
                let pyro_tensor =
                    Tensor::from_vec(pyro_embed, (1, token_count, hidden_dim), model.device())?;
                let pyro_states = model.vision_model().encode_hidden_with_states(
                    &pyro_tensor,
                    patch,
                    model.config().use_3d_rope,
                    model.device(),
                )?;
                for (idx, state_tensor) in pyro_states.into_iter().enumerate() {
                    let expected_slice = expected_states.slice(s![idx, offset..end, ..]);
                    let label = format!(
                        "{} encoder_from_python_image{}_layer{}",
                        case.name, img_idx, idx
                    );
                    if allow_state_drift {
                        let stage = state_tensor.to_dtype(DType::F32)?.contiguous()?;
                        let elem = stage.shape().elem_count();
                        let actual = stage.reshape((elem,))?.to_vec1::<f32>()?;
                        let expected_flat = expected_slice
                            .as_slice()
                            .expect("py input state slice contiguous");
                        let mut max_diff = 0f32;
                        for (a, b) in actual.iter().zip(expected_flat.iter()) {
                            max_diff = max_diff.max((a - b).abs());
                        }
                        eprintln!("{label} max abs diff {}", max_diff);
                    } else {
                        assert_close(
                            &state_tensor,
                            expected_slice
                                .as_slice()
                                .expect("py input state slice contiguous"),
                            5e-3,
                            &label,
                        )?;
                    }
                }
                offset = end;
            }

            if case.name == "sample_doc" && patches.len() == 1 {
                let debug_path = asset_path("outputs/sample_doc_layer17_debug.npz");
                if debug_path.exists() {
                    eprintln!(
                        "sample_doc layer17 debug instrumentation active at {}",
                        debug_path.display()
                    );
                    let mut debug_reader = NpzReader::new(File::open(&debug_path)?)
                        .context("failed to read layer17 debug npz")?;
                    let debug_norm1: Array2<f32> = debug_reader
                        .by_name("norm1")
                        .context("layer17 debug missing norm1")?;
                    let debug_attn: Array2<f32> = debug_reader
                        .by_name("attn_out")
                        .context("layer17 debug missing attn_out")?;
                    let debug_after_attn: Array2<f32> = debug_reader
                        .by_name("after_attn")
                        .context("layer17 debug missing after_attn")?;
                    let debug_norm2: Array2<f32> = debug_reader
                        .by_name("norm2")
                        .context("layer17 debug missing norm2")?;
                    let debug_mlp: Array2<f32> = debug_reader
                        .by_name("mlp_out")
                        .context("layer17 debug missing mlp_out")?;
                    let debug_output: Array2<f32> = debug_reader
                        .by_name("output")
                        .context("layer17 debug missing output")?;

                    let layer_index = 17usize;
                    let patch = &patches[0];
                    let token_count = patch.grid_thw.0 * patch.grid_thw.1 * patch.grid_thw.2;
                    let layer_input = expected_states.slice(s![layer_index, 0..token_count, ..]);
                    let pyro_vec: Vec<f32> = layer_input.iter().copied().collect();
                    let pyro_tensor =
                        Tensor::from_vec(pyro_vec, (1, token_count, hidden_dim), model.device())?;
                    let debug = model.vision_model().debug_layer_outputs(
                        &pyro_tensor,
                        patch,
                        layer_index,
                        model.config().use_3d_rope,
                        model.device(),
                    )?;
                    let report_stage =
                        |tensor: &Tensor, expected: &Array2<f32>, label: &str| -> Result<()> {
                            let stage = tensor
                                .to_dtype(DType::F32)?
                                .reshape((token_count, hidden_dim))?
                                .contiguous()?;
                            let elem = stage.shape().elem_count();
                            let actual = stage.reshape((elem,))?.to_vec1::<f32>()?;
                            let expected_slice = expected
                                .as_slice()
                                .context("layer17 debug slice not contiguous")?;
                            let mut max_diff = 0f32;
                            for (a, b) in actual.iter().zip(expected_slice.iter()) {
                                max_diff = max_diff.max((a - b).abs());
                            }
                            eprintln!("sample_doc debug {} max diff {}", label, max_diff);
                            Ok(())
                        };
                    report_stage(&debug.norm1, &debug_norm1, "layer17 norm1")?;
                    report_stage(&debug.attn_out, &debug_attn, "layer17 attn_out")?;
                    report_stage(&debug.after_attn, &debug_after_attn, "layer17 after_attn")?;
                    report_stage(&debug.norm2, &debug_norm2, "layer17 norm2")?;
                    report_stage(&debug.mlp_out, &debug_mlp, "layer17 mlp_out")?;
                    report_stage(&debug.output, &debug_output, "layer17 output")?;
                } else {
                    eprintln!(
                        "sample_doc layer17 debug npz missing at {}",
                        debug_path.display()
                    );
                }
            }
        }

        if case.name == "sample_doc" {
            let py_pos_path = Path::new("outputs/py_pos_embed.npy");
            if py_pos_path.exists() && !patches.is_empty() {
                let py_pos: Array2<f32> = ndarray_npy::read_npy(py_pos_path)
                    .context("failed to read python positional embedding dump")?;
                let candle_pos = model
                    .vision_model()
                    .debug_positional_encoding(patches[0].grid_thw, model.device())?
                    .to_dtype(DType::F32)?
                    .contiguous()?;
                assert_close(
                    &candle_pos,
                    py_pos
                        .as_slice()
                        .expect("python positional embed contiguous"),
                    5e-3,
                    "siglip positional embedding (sample_doc)",
                )?;
            }
        }

        assert_close(
            &siglip_tensor,
            siglip_hidden_np
                .as_slice()
                .expect("siglip array contiguous"),
            5e-3,
            &format!("{} siglip_hidden", case.name),
        )?;

        let projector_tensor = concat_tensors(projector_chunks)?.contiguous()?;
        assert_close(
            &projector_tensor,
            projector_np.as_slice().expect("projector array contiguous"),
            5e-3,
            &format!("{} projector_embeddings", case.name),
        )?;

        let grids = fixture_grids.clone();
        let (prompt_tokens, image_mask_vec) =
            build_prompt_tokens(tokenizer, &metadata.prompt, &grids, model.config())?;
        let fixture_tokens: Vec<i64> = input_ids_np.iter().copied().collect();
        assert_eq!(
            prompt_tokens, fixture_tokens,
            "{} prompt tokens diverged",
            case.name
        );

        let seq_len = fixture_tokens.len();
        let vocab = logits_np.shape()[1];
        let input_ids_tensor =
            Tensor::from_vec(fixture_tokens.clone(), (1, seq_len), model.device())?
                .to_dtype(DType::I64)?;
        let attention_mask_vec: Vec<u8> = attention_mask_np
            .iter()
            .map(|&v| u8::try_from(v).unwrap_or(0))
            .collect();
        let attention_mask_tensor =
            Tensor::from_vec(attention_mask_vec.clone(), (1, seq_len), model.device())?
                .to_dtype(DType::U8)?;

        let base_embeddings =
            gather_token_embeddings(model.decoder().embed_tokens(), &input_ids_tensor)?;
        let image_mask_tensor =
            Tensor::from_vec(image_mask_vec.clone(), (1, seq_len), model.device())?
                .to_dtype(DType::U8)?;
        let fused = inject_image_embeddings(
            &base_embeddings,
            &image_mask_tensor,
            &[projector_tensor.clone()],
        )?
        .to_dtype(DType::F32)?;
        assert_close(
            &fused,
            fused_np.as_slice().expect("fused array contiguous"),
            5e-3,
            &format!("{} fused_embeddings", case.name),
        )?;

        let image_grid_metadata = vec![grids.clone()];
        let (position_ids, rope_deltas) = compute_position_ids(
            model.config(),
            &input_ids_tensor,
            Some(&attention_mask_tensor),
            &image_grid_metadata,
        )?;
        assert_int_match(
            &position_ids,
            position_ids_np.as_slice().expect("position ids contiguous"),
            &format!("{} position_ids", case.name),
        )?;
        assert_int_match(
            &rope_deltas,
            rope_deltas_np.as_slice().expect("rope deltas contiguous"),
            &format!("{} rope_deltas", case.name),
        )?;

        let decoder = model.decoder();
        let prefill = decoder.forward(
            None,
            Some(&fused),
            Some(&attention_mask_tensor),
            Some(&position_ids),
            None,
            false,
        )?;
        let logits = prefill
            .logits
            .get(0)?
            .get(seq_len - 1)?
            .to_dtype(DType::F32)?;
        assert_eq!(
            logits.shape().dims1()?,
            vocab,
            "logit dimension mismatch for {}",
            case.name
        );
        assert_close(
            &logits,
            logits_np.as_slice().expect("logits contiguous"),
            1e-2,
            &format!("{} next_token_logits", case.name),
        )?;

        Ok(())
    }

    fn concat_tensors(mut tensors: Vec<Tensor>) -> Result<Tensor> {
        ensure!(!tensors.is_empty(), "cannot concatenate empty tensor list");
        if tensors.len() == 1 {
            Ok(tensors.pop().expect("length checked above"))
        } else {
            let refs: Vec<&Tensor> = tensors.iter().collect();
            Ok(Tensor::cat(&refs, 0)?)
        }
    }

    fn assert_close(tensor: &Tensor, expected: &[f32], tol: f32, label: &str) -> Result<()> {
        let elem = tensor.shape().elem_count();
        let actual = tensor.reshape((elem,))?.to_vec1::<f32>()?;
        ensure!(
            actual.len() == expected.len(),
            "{} length mismatch ({} vs {})",
            label,
            actual.len(),
            expected.len()
        );
        let mut max_diff = 0f32;
        for (a, b) in actual.iter().zip(expected.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff <= tol,
            "{} max abs diff {} exceeds tolerance {}",
            label,
            max_diff,
            tol
        );
        Ok(())
    }

    fn assert_int_match(tensor: &Tensor, expected: &[i64], label: &str) -> Result<()> {
        let elem = tensor.shape().elem_count();
        let actual = tensor.reshape((elem,))?.to_vec1::<i64>()?;
        ensure!(
            actual == expected,
            "{} mismatch between tensor ({:?}...) and fixture ({:?}...)",
            label,
            &actual[..actual.len().min(4)],
            &expected[..expected.len().min(4)]
        );
        Ok(())
    }
}
