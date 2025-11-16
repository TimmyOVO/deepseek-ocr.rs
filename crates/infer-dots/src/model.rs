use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::DynamicImage;
use serde::Deserialize;
use tokenizers::Tokenizer;

use deepseek_ocr_core::sampling::{init_rng, select_token_id};
use deepseek_ocr_core::tensor::gather_token_embeddings;
use deepseek_ocr_core::{
    DecodeOutcome, DecodeParameters, ModelKind, ModelLoadArgs, OcrEngine, VisionSettings,
    normalize_text,
};

use crate::{
    config::{DotsOcrConfig, load_dots_config},
    tokenizer::DotsImageTokens,
    transformer::Qwen2LanguageModel,
    vision::{
        DotsVisionModel,
        preprocess::{DotsPreprocessConfig, preprocess_images},
    },
};

const DEFAULT_WEIGHTS_INDEX: &str = "dots.ocr/model.safetensors.index.json";
const FALLBACK_EOS_TOKEN: &str = "<|endoftext|>";
const IMAGE_PLACEHOLDER: &str = "<image>";
const MAX_INLINE_INDEX_BYTES: u64 = 64 * 1024 * 1024;

pub struct DotsOcrModel {
    device: Device,
    dtype: DType,
    config: Arc<DotsOcrConfig>,
    preprocess: DotsPreprocessConfig,
    weights_path: PathBuf,
    vision: DotsVisionModel,
    decoder: Qwen2LanguageModel,
}

impl DotsOcrModel {
    pub fn load(args: ModelLoadArgs<'_>) -> Result<Self> {
        if args.kind != ModelKind::DotsOcr {
            bail!(
                "ModelKind::{:?} cannot be loaded by the Dots OCR engine",
                args.kind
            );
        }

        let config = Arc::new(load_dots_config(args.config_path)?);
        let preprocess = DotsPreprocessConfig::load(None)?;
        let weights_path = args
            .weights_path
            .map(Path::to_path_buf)
            .unwrap_or_else(default_weights_index_path);
        let shard_paths = resolve_weight_paths(&weights_path)?;
        let shard_refs: Vec<_> = shard_paths.iter().map(|path| path.as_path()).collect();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shard_refs, args.dtype, &args.device)
                .with_context(|| {
                    format!(
                        "failed to mmap dots.ocr weights from {}",
                        weights_path.display()
                    )
                })?
        };
        let vision_cfg = Arc::new(config.vision.clone());
        let decoder_cfg = Arc::new(config.text.clone());
        let vision = DotsVisionModel::load(Arc::clone(&vision_cfg), &vb.pp("vision_tower"))
            .context("failed to load Dots vision tower")?;
        let decoder = Qwen2LanguageModel::load(Arc::clone(&decoder_cfg), &vb)
            .context("failed to load Dots text decoder")?;

        Ok(Self {
            device: args.device,
            dtype: args.dtype,
            config,
            preprocess,
            weights_path,
            vision,
            decoder,
        })
    }

    fn encode_images(&self, pixel_values: &Tensor, grids: &[[u32; 3]]) -> Result<Vec<Tensor>> {
        if grids.is_empty() {
            return Ok(Vec::new());
        }
        // Pre-processing always emits F32 tensors; convert them to the model dtype so
        // downstream convolutions (which may be BF16) can run without dtype mismatches.
        let vision_input = if pixel_values.dtype() == self.dtype {
            pixel_values.clone()
        } else {
            pixel_values.to_dtype(self.dtype)?
        };
        let vision_hidden = self.vision.forward(&vision_input, grids)?;
        let mut outputs = Vec::with_capacity(grids.len());
        let mut offset = 0usize;
        let merge = self.config.vision.spatial_merge_size;
        let total = vision_hidden.dim(0)? as usize;
        for (idx, grid) in grids.iter().enumerate() {
            let tokens = vision_token_count(*grid, merge)?;
            ensure!(
                offset + tokens <= total,
                "vision output truncated for image {idx}: need {} tokens, have {}",
                tokens,
                total.saturating_sub(offset)
            );
            let chunk = vision_hidden
                .narrow(0, offset, tokens)?
                .contiguous()
                .context("vision chunk must be contiguous")?;
            outputs.push(chunk);
            offset += tokens;
        }
        ensure!(
            offset == total,
            "vision tokens consumed ({offset}) do not match encoder output ({total})"
        );
        Ok(outputs)
    }

    fn prepare_prompt(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        grids: &[[u32; 3]],
        vision_embeddings: Vec<Tensor>,
        image_tokens: &DotsImageTokens,
    ) -> Result<PreparedPrompt> {
        let merge = self.config.vision.spatial_merge_size;
        ensure!(
            grids.len() == vision_embeddings.len(),
            "image metadata ({}) mismatches encoded embeddings ({})",
            grids.len(),
            vision_embeddings.len()
        );
        let per_image_tokens: Vec<usize> = grids
            .iter()
            .map(|grid| vision_token_count(*grid, merge))
            .collect::<Result<_>>()?;
        let (input_tokens, image_mask_vec) =
            build_prompt_inputs(tokenizer, prompt, &per_image_tokens, image_tokens)?;
        ensure!(
            !input_tokens.is_empty(),
            "prompt must produce at least one token"
        );
        let placeholder_count = image_mask_vec.iter().filter(|&&flag| flag != 0).count();
        let mut available_tokens = 0usize;
        for tensor in &vision_embeddings {
            available_tokens += tensor.dim(0)?;
        }
        ensure!(
            placeholder_count == available_tokens,
            "image placeholder span ({placeholder_count}) mismatches vision outputs ({available_tokens})"
        );

        let device = &self.device;
        let prompt_len = input_tokens.len();
        let input_ids = Tensor::from_vec(input_tokens.clone(), (1, prompt_len), device)?
            .to_dtype(DType::I64)?;
        let base_embeddings = gather_token_embeddings(self.decoder.embed_tokens(), &input_ids)?;
        let image_mask = Tensor::from_vec(image_mask_vec.clone(), (1, prompt_len), device)?
            .to_dtype(DType::U8)?;

        let fused_embeddings = if placeholder_count == 0 {
            base_embeddings
        } else {
            let replacements = flatten_vision_embeddings(vision_embeddings)?
                .expect("placeholder count validated against embeddings");
            let per_batch = vec![replacements];
            inject_image_embeddings(&base_embeddings, &image_mask, &per_batch)?
        };

        Ok(PreparedPrompt {
            embeddings: fused_embeddings,
            context_tokens: input_tokens,
        })
    }
}

impl OcrEngine for DotsOcrModel {
    fn kind(&self) -> ModelKind {
        ModelKind::DotsOcr
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn weights_path(&self) -> Option<&Path> {
        Some(self.weights_path.as_path())
    }

    fn decode(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        images: &[DynamicImage],
        _vision: VisionSettings,
        params: &DecodeParameters,
        stream: Option<&dyn Fn(usize, &[i64])>,
    ) -> Result<DecodeOutcome> {
        ensure!(
            params.use_cache,
            "Dots OCR currently requires use_cache=true for decoding"
        );
        let image_tokens = DotsImageTokens::resolve(tokenizer, &self.config)?;
        let (pixel_values, grids) = preprocess_images(images, &self.device, &self.preprocess)?;
        let vision_embeddings = if grids.is_empty() {
            Vec::new()
        } else {
            self.encode_images(&pixel_values, &grids)?
        };
        let prepared =
            self.prepare_prompt(tokenizer, prompt, &grids, vision_embeddings, &image_tokens)?;
        let prompt_len = prepared.len();
        if params.max_new_tokens == 0 {
            return Ok(DecodeOutcome {
                text: String::new(),
                prompt_tokens: prompt_len,
                response_tokens: 0,
                generated_tokens: Vec::new(),
            });
        }

        let eos_token_id = resolve_eos_token_id(tokenizer);
        let PreparedPrompt {
            embeddings,
            mut context_tokens,
        } = prepared;
        let mut cache = self.decoder.new_cache();
        let mut guard = self.decoder.prompt_guard(&mut cache);
        let prefill = self.decoder.forward(
            None,
            Some(&embeddings),
            None,
            None,
            Some(guard.cache()),
            params.use_cache,
        )?;
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
            let decode = self.decoder.forward(
                None,
                Some(&decode_embeddings),
                None,
                None,
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

struct PreparedPrompt {
    embeddings: Tensor,
    context_tokens: Vec<i64>,
}

impl PreparedPrompt {
    fn len(&self) -> usize {
        self.context_tokens.len()
    }
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

fn default_weights_index_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(DEFAULT_WEIGHTS_INDEX)
}

fn resolve_weight_paths(path: &Path) -> Result<Vec<PathBuf>> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    {
        return load_index_paths(path);
    }
    ensure!(
        path.exists(),
        "weights file {} does not exist",
        path.display()
    );
    if let Some(paths) = try_load_inline_index(path)? {
        return Ok(paths);
    }
    Ok(vec![path.to_path_buf()])
}

fn load_index_paths(index_path: &Path) -> Result<Vec<PathBuf>> {
    let bytes = fs::read(index_path).with_context(|| {
        format!(
            "failed to read safetensors index from {}",
            index_path.display()
        )
    })?;
    parse_index_bytes(index_path, bytes)
}

fn try_load_inline_index(path: &Path) -> Result<Option<Vec<PathBuf>>> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to fetch metadata for weights at {}", path.display()))?;
    if metadata.len() > MAX_INLINE_INDEX_BYTES {
        return Ok(None);
    }
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read candidate index file {}", path.display()))?;
    match parse_index_bytes(path, bytes) {
        Ok(paths) => Ok(Some(paths)),
        Err(_) => Ok(None),
    }
}

fn parse_index_bytes(index_path: &Path, bytes: Vec<u8>) -> Result<Vec<PathBuf>> {
    let index: SafetensorsIndex = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "failed to parse safetensors index at {}",
            index_path.display()
        )
    })?;
    ensure!(
        !index.weight_map.is_empty(),
        "safetensors index {} did not list any shards",
        index_path.display()
    );
    let base = index_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut shards = BTreeSet::new();
    for entry in index.weight_map.values() {
        let candidate = if Path::new(entry).is_absolute() {
            PathBuf::from(entry)
        } else {
            base.join(entry)
        };
        ensure!(
            candidate.exists(),
            "safetensors shard {} referenced by {} is missing",
            candidate.display(),
            index_path.display()
        );
        shards.insert(candidate);
    }
    Ok(shards.into_iter().collect())
}

fn vision_token_count(grid: [u32; 3], merge: usize) -> Result<usize> {
    ensure!(merge > 0, "spatial merge size must be positive");
    let t = grid[0] as usize;
    let h = grid[1] as usize;
    let w = grid[2] as usize;
    ensure!(
        h % merge == 0 && w % merge == 0,
        "grid {}x{} not divisible by merge {}",
        h,
        w,
        merge
    );
    Ok(t * (h / merge) * (w / merge))
}

fn build_prompt_inputs(
    tokenizer: &Tokenizer,
    prompt: &str,
    per_image_tokens: &[usize],
    image_tokens: &DotsImageTokens,
) -> Result<(Vec<i64>, Vec<u8>)> {
    let segments: Vec<&str> = prompt.split(IMAGE_PLACEHOLDER).collect();
    ensure!(
        segments.len().saturating_sub(1) == per_image_tokens.len(),
        "prompt/image mismatch: {} slots vs {} images",
        segments.len().saturating_sub(1),
        per_image_tokens.len()
    );
    let mut tokens = Vec::new();
    let mut mask = Vec::new();
    for (idx, segment) in segments.iter().enumerate() {
        if !segment.is_empty() {
            let encoding = tokenizer
                .encode(*segment, false)
                .map_err(|err| anyhow!("tokenization failed: {err}"))?;
            tokens.extend(encoding.get_ids().iter().map(|&id| id as i64));
            mask.extend(std::iter::repeat(0u8).take(encoding.len()));
        }
        if idx < per_image_tokens.len() {
            let placeholders = per_image_tokens[idx];
            ensure!(placeholders > 0, "image {idx} produced zero vision tokens");
            tokens.push(image_tokens.start as i64);
            mask.push(0);
            tokens.extend(std::iter::repeat(image_tokens.pad as i64).take(placeholders));
            mask.extend(std::iter::repeat(1u8).take(placeholders));
            tokens.push(image_tokens.end as i64);
            mask.push(0);
        }
    }
    Ok((tokens, mask))
}

fn flatten_vision_embeddings(mut per_image: Vec<Tensor>) -> Result<Option<Tensor>> {
    if per_image.is_empty() {
        return Ok(None);
    }
    if per_image.len() == 1 {
        return Ok(per_image.pop());
    }
    let refs: Vec<&Tensor> = per_image.iter().collect();
    Ok(Some(Tensor::cat(&refs, 0)?))
}

fn inject_image_embeddings(
    embeddings: &Tensor,
    mask: &Tensor,
    per_batch: &[Tensor],
) -> Result<Tensor> {
    let (batch, seq_len, hidden) = embeddings.shape().dims3()?;
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
            .with_context(|| format!("missing image embeddings for batch {b}"))?
            .to_dtype(row.dtype())?
            .to_device(row.device())?;
        let (rep_tokens, _) = replacements.shape().dims2()?;
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
            "not all image embeddings were consumed ({rep_offset} of {ones})"
        );
        let refs: Vec<&Tensor> = segments.iter().collect();
        rows.push(Tensor::cat(&refs, 0)?);
    }
    let refs: Vec<&Tensor> = rows.iter().collect();
    Ok(Tensor::stack(&refs, 0)?)
}

fn resolve_eos_token_id(tokenizer: &Tokenizer) -> Option<i64> {
    tokenizer
        .token_to_id(FALLBACK_EOS_TOKEN)
        .map(|id| id as i64)
}

fn single_token_tensor(token: i64, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(vec![token], (1, 1), device)?.to_dtype(DType::I64)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../dots.ocr/tokenizer.json")
    }

    #[test]
    fn prompt_builder_inserts_image_tokens() -> Result<()> {
        let tokenizer =
            Tokenizer::from_file(tokenizer_path()).expect("dots.ocr tokenizer should load");
        let cfg = Arc::new(load_dots_config(None)?);
        let tokens = DotsImageTokens::resolve(&tokenizer, &cfg)?;
        let counts = vec![4usize];
        let (ids, mask) =
            build_prompt_inputs(&tokenizer, "User: <image> Answer:", &counts, &tokens)?;
        assert_eq!(mask.len(), ids.len());
        let pad_id = tokens.pad as i64;
        let pad_positions: Vec<_> = ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &id)| (id == pad_id).then_some(idx))
            .collect();
        assert_eq!(pad_positions.len(), 4);
        for idx in pad_positions {
            assert_eq!(mask[idx], 1);
        }
        Ok(())
    }

    #[test]
    fn vision_token_count_respects_merge() -> Result<()> {
        let tokens = vision_token_count([1, 4, 4], 2)?;
        assert_eq!(tokens, 4);
        Ok(())
    }
}
