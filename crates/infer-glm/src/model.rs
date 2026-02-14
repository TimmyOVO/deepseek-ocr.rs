use std::{
    convert::TryFrom,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, anyhow, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::DynamicImage;
#[cfg(feature = "trace-logits")]
use std::cmp::Ordering;
use tokenizers::Tokenizer;

use crate::{
    config::{
        GlmGenerationConfig, GlmOcrConfig, GlmPreprocessorConfig, load_config,
        load_generation_config, load_preprocessor,
    },
    transformer::GlmTextDecoder,
    vision::{GlmVisionModel, preprocess_images},
};

use deepseek_ocr_core::{
    benchmark::Timer,
    build_prompt_tokens_with,
    grid_token_count,
    inference::{
        DecodeOutcome, DecodeParameters, ModelKind, ModelLoadArgs, OcrEngine, VisionSettings,
    },
    normalize_text,
    PromptBuildOptions,
    PromptTokenSequence,
    sampling::{init_rng, select_token_id},
    tensor::{inject_embeddings_by_mask, into_dtype_if_needed, gather_token_embeddings},
};

pub const DEFAULT_WEIGHTS_PATH: &str = "GLM-OCR/model.safetensors";

const TOKEN_GMASK: i64 = 59248;
const TOKEN_SOP: i64 = 59250;
const TOKEN_USER: i64 = 59253;
const TOKEN_ASSISTANT: i64 = 59254;
const TOKEN_NEWLINE: i64 = 10;

pub struct GlmOcrModel {
    config: Arc<GlmOcrConfig>,
    preprocessor: Arc<GlmPreprocessorConfig>,
    generation_config: GlmGenerationConfig,
    device: Device,
    dtype: DType,
    weights_path: PathBuf,
    vision: GlmVisionModel,
    decoder: GlmTextDecoder,
}

impl GlmOcrModel {
    pub fn load(args: &ModelLoadArgs<'_>) -> Result<Self> {
        let loaded = load_config(args.config_path)?;
        let preprocessor = load_preprocessor(&loaded.path)?;
        let generation_config = load_generation_config(&loaded.path)?.unwrap_or_default();
        let resolved_weights = args
            .weights_path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_WEIGHTS_PATH));
        ensure!(
            resolved_weights.exists(),
            "weights not found at {}",
            resolved_weights.display()
        );

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[resolved_weights.as_path()],
                args.dtype,
                &args.device,
            )
        }
        .with_context(|| {
            format!(
                "failed to mmap GLM weights at {}",
                resolved_weights.display()
            )
        })?;

        let config = Arc::new(loaded.value);
        let vision = GlmVisionModel::load(&vb, Arc::new(config.vision_config.clone()), args.dtype)
            .context("failed to load GLM vision tower")?;
        let decoder = GlmTextDecoder::load(Arc::new(config.text_config.clone()), &vb)
            .context("failed to load GLM text decoder")?;

        Ok(Self {
            config,
            preprocessor: Arc::new(preprocessor.value),
            generation_config,
            device: args.device.clone(),
            dtype: args.dtype,
            weights_path: resolved_weights,
            vision,
            decoder,
        })
    }

    fn effective_eos_ids(&self) -> Vec<i64> {
        if !self.generation_config.eos_token_id.is_empty() {
            return self.generation_config.eos_token_id.clone();
        }
        self.config.text_config.eos_token_id.clone()
    }

    fn validate_decode_params(&self, params: &DecodeParameters) -> Result<()> {
        ensure!(!params.do_sample, "GLM backend requires do_sample=false");
        ensure!(
            params.temperature == 0.0,
            "GLM backend requires temperature=0.0"
        );
        if let Some(sample) = self.generation_config.do_sample {
            ensure!(!sample, "generation_config.do_sample must be false");
        }
        Ok(())
    }

    fn build_prompt_tokens(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        image_grids: &[(usize, usize, usize)],
    ) -> Result<PromptTokenSequence> {
        let prefix = [TOKEN_GMASK, TOKEN_SOP, TOKEN_USER, TOKEN_NEWLINE];
        let suffix = [TOKEN_ASSISTANT, TOKEN_NEWLINE];
        let merge = self.preprocessor.spatial_merge_size;
        let sequence = build_prompt_tokens_with(
            tokenizer,
            prompt,
            image_grids.len(),
            PromptBuildOptions::image_slots("images")
                .with_prefix(&prefix)
                .with_suffix(&suffix),
            |idx, tokens, mask| {
                let image_token_count = grid_token_count(image_grids[idx], merge)?;
                tokens.push(self.config.image_start_token_id);
                mask.push(0);
                tokens.extend(std::iter::repeat_n(
                    self.config.image_token_id,
                    image_token_count,
                ));
                mask.extend(std::iter::repeat_n(1u8, image_token_count));
                tokens.push(self.config.image_end_token_id);
                mask.push(0);
                Ok(())
            },
        )?;
        Ok(sequence)
    }

    fn build_position_ids(
        &self,
        input_ids: &[i64],
        image_grids: &[(usize, usize, usize)],
    ) -> Result<(Tensor, i64)> {
        let merge = self.config.vision_config.spatial_merge_size;
        let mut token_types = Vec::with_capacity(input_ids.len());
        let mut in_video = false;
        for &token in input_ids {
            if token == self.config.video_start_token_id {
                in_video = true;
            } else if token == self.config.video_end_token_id {
                in_video = false;
            }
            if token == self.config.image_token_id && !in_video {
                token_types.push(TokenType::Image);
            } else if token == self.config.image_token_id && in_video {
                token_types.push(TokenType::Video);
            } else {
                token_types.push(TokenType::Text);
            }
        }

        let mut groups = Vec::new();
        let mut start = 0usize;
        while start < token_types.len() {
            let ty = token_types[start];
            let mut end = start + 1;
            while end < token_types.len() && token_types[end] == ty {
                end += 1;
            }
            groups.push((ty, start, end));
            start = end;
        }

        let seq_len = input_ids.len();
        let mut t_axis = Vec::with_capacity(seq_len);
        let mut h_axis = Vec::with_capacity(seq_len);
        let mut w_axis = Vec::with_capacity(seq_len);
        let mut max_position = -1i64;
        let mut image_index = 0usize;
        let mut video_index = 0usize;
        let mut video_group_index = 0usize;
        let mut video_frame_num = 1usize;

        for (ty, start, end) in groups {
            let st_idx = max_position + 1;
            match ty {
                TokenType::Image => {
                    let (t, h, w) = image_grids
                        .get(image_index)
                        .copied()
                        .ok_or_else(|| anyhow!("not enough image grids for image tokens"))?;
                    let llm_t = t;
                    let llm_h = h / merge;
                    let llm_w = w / merge;
                    for t_idx in 0..llm_t {
                        for h_idx in 0..llm_h {
                            for w_idx in 0..llm_w {
                                let t_pos = st_idx + t_idx as i64;
                                let h_pos = st_idx + h_idx as i64;
                                let w_pos = st_idx + w_idx as i64;
                                t_axis.push(t_pos);
                                h_axis.push(h_pos);
                                w_axis.push(w_pos);
                                max_position = max_position.max(t_pos).max(h_pos).max(w_pos);
                            }
                        }
                    }
                    image_index += 1;
                    video_frame_num = 1;
                }
                TokenType::Video => {
                    let (t, h, w) = image_grids
                        .get(video_index)
                        .copied()
                        .ok_or_else(|| anyhow!("not enough video grids"))?;
                    let llm_t = video_frame_num;
                    let llm_h = h / merge;
                    let llm_w = w / merge;
                    for t_idx in 0..llm_t {
                        for h_idx in 0..llm_h {
                            for w_idx in 0..llm_w {
                                let t_pos = st_idx + t_idx as i64;
                                let h_pos = st_idx + h_idx as i64;
                                let w_pos = st_idx + w_idx as i64;
                                t_axis.push(t_pos);
                                h_axis.push(h_pos);
                                w_axis.push(w_pos);
                                max_position = max_position.max(t_pos).max(h_pos).max(w_pos);
                            }
                        }
                    }
                    video_group_index += 1;
                    if video_group_index >= t {
                        video_index += 1;
                        video_group_index = 0;
                    }
                    video_frame_num += 1;
                }
                TokenType::Text => {
                    let len = end - start;
                    for i in 0..len {
                        let val = st_idx + i as i64;
                        t_axis.push(val);
                        h_axis.push(val);
                        w_axis.push(val);
                        max_position = max_position.max(val);
                    }
                    video_frame_num = 1;
                }
            }
        }

        ensure!(
            t_axis.len() == input_ids.len(),
            "position ids length {} != input length {}",
            t_axis.len(),
            input_ids.len()
        );

        let t_tensor = Tensor::from_vec(t_axis, (1, seq_len), &self.device)?;
        let h_tensor = Tensor::from_vec(h_axis, (1, seq_len), &self.device)?;
        let w_tensor = Tensor::from_vec(w_axis, (1, seq_len), &self.device)?;
        let position_ids =
            Tensor::stack(&[t_tensor, h_tensor, w_tensor], 0)?.to_dtype(DType::I64)?;

        let max_position = max_position.max(0);
        let rope_delta = max_position + 1 - seq_len as i64;
        let next_position_base = rope_delta + seq_len as i64;
        Ok((position_ids, next_position_base))
    }

    fn prepare_inputs(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        images: &[DynamicImage],
        vision: VisionSettings,
    ) -> Result<PreparedInputs> {
        let prepare_timer = Timer::new("vision.prepare_inputs");
        let image_batch = preprocess_images(images, &self.preprocessor, &self.device, vision)
            .context("failed to preprocess GLM images")?;

        #[cfg(feature = "trace-logits")]
        {
            if std::env::var_os("GLM_DEBUG_VISION").is_some() {
                let (rows, cols) = image_batch.pixel_values.shape().dims2()?;
                eprintln!(
                    "[glm-debug] pixel_values shape=({rows},{cols}) dtype={:?}",
                    image_batch.pixel_values.dtype()
                );
                debug_tensor_stats("pixel_values", &image_batch.pixel_values)?;
            }
        }

        let vision_embeds = if image_batch.image_grid_thw.is_empty() {
            Tensor::zeros(
                (0, self.config.vision_config.out_hidden_size),
                self.dtype,
                &self.device,
            )?
        } else {
            let vision_timer = Timer::new("vision.compute_embeddings");
            let vision_tokens = self
                .vision
                .encode(&image_batch.pixel_values, &image_batch.image_grid_thw)
                .context("GLM vision encoder failed")?;
            vision_timer.finish(|event| {
                event.add_field("images", image_batch.image_grid_thw.len() as u64);
            });
            into_dtype_if_needed(vision_tokens, self.dtype)?
        };

        #[cfg(feature = "trace-logits")]
        {
            if std::env::var_os("GLM_DEBUG_VISION").is_some() {
                let (rows, cols) = vision_embeds.shape().dims2()?;
                eprintln!(
                    "[glm-debug] vision_embeds shape=({rows},{cols}) dtype={:?}",
                    vision_embeds.dtype()
                );
                debug_tensor_stats("vision_embeds", &vision_embeds)?;
            }
        }

        let prompt_timer = Timer::new("prompt.build_tokens");
        let prompt_sequence = self.build_prompt_tokens(tokenizer, prompt, &image_batch.image_grid_thw)?;
        prompt_timer.finish(|event| {
            event.add_field("tokens", prompt_sequence.tokens.len() as u64);
            event.add_field("images", image_batch.image_grid_thw.len() as u64);
        });

        let placeholder_tokens = prompt_sequence.image_token_count();
        let PromptTokenSequence {
            tokens: prompt_tokens,
            image_mask,
        } = prompt_sequence;

        let prompt_len = prompt_tokens.len();
        ensure!(prompt_len > 0, "prompt must contain at least one token");

        let input_ids = Tensor::from_vec(prompt_tokens.clone(), (1, prompt_len), &self.device)?
            .to_dtype(DType::I64)?;
        let mut embeddings = gather_token_embeddings(self.decoder.embed_tokens(), &input_ids)?;

        if placeholder_tokens > 0 {
            let (available, _) = vision_embeds.shape().dims2()?;
            ensure!(
                placeholder_tokens == available,
                "image placeholder span ({placeholder_tokens}) mismatches vision embeds ({available})"
            );
            let mask = Tensor::from_vec(image_mask, (1, prompt_len), &self.device)?
                .to_dtype(DType::U8)?;
            embeddings = inject_embeddings_by_mask(
                &embeddings,
                &mask,
                std::slice::from_ref(&vision_embeds),
            )
                .context("failed to inject image embeddings into prompt")?;
        }

        let attention_mask = Tensor::ones((1, prompt_len), DType::U8, &self.device)?;
        let (position_ids, next_position_base) =
            self.build_position_ids(&prompt_tokens, &image_batch.image_grid_thw)?;

        prepare_timer.finish(|event| {
            event.add_field("prompt_tokens", prompt_len as u64);
            event.add_field("images", image_batch.image_grid_thw.len() as u64);
        });

        Ok(PreparedInputs {
            embeddings,
            attention_mask,
            position_ids,
            prompt_tokens,
            next_position_base,
            prompt_len,
        })
    }
}

#[cfg(feature = "trace-logits")]
fn debug_tensor_stats(name: &str, tensor: &Tensor) -> Result<()> {
    let data = tensor.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    if data.is_empty() || data[0].is_empty() {
        eprintln!("[glm-debug] {name} empty");
        return Ok(());
    }

    let rows = data.len();
    let cols = data[0].len();
    let sample = &data[0];
    let first8 = sample.iter().copied().take(8).collect::<Vec<_>>();
    let row_sum: f64 = sample.iter().map(|&v| v as f64).sum();
    let row_l2: f64 = sample
        .iter()
        .map(|&v| {
            let x = v as f64;
            x * x
        })
        .sum::<f64>()
        .sqrt();

    let mut global_min = f32::INFINITY;
    let mut global_max = f32::NEG_INFINITY;
    let mut global_sum = 0f64;
    let mut global_sumsq = 0f64;
    let mut global_count = 0usize;
    let mut hash: u64 = 0xcbf29ce484222325;

    let probe_indices: [usize; 20] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10_000, 100_000, 1_000_000, 2_000_000,
        3_000_000, 5_000_000, 7_000_000,
    ];
    let mut probes = Vec::new();
    let dump_dir = std::env::var_os("GLM_DEBUG_DUMP_DIR").map(PathBuf::from);
    let mut flat_dump = dump_dir
        .as_ref()
        .map(|_| Vec::<f32>::with_capacity(rows * cols));

    let mut flat_idx = 0usize;
    for row in &data {
        for &v in row {
            global_min = global_min.min(v);
            global_max = global_max.max(v);
            let x = v as f64;
            global_sum += x;
            global_sumsq += x * x;
            global_count += 1;
            hash ^= v.to_bits() as u64;
            hash = hash.wrapping_mul(0x100000001b3);

            if probe_indices.contains(&flat_idx) {
                probes.push((flat_idx, v, v.to_bits()));
            }
            if let Some(values) = flat_dump.as_mut() {
                values.push(v);
            }
            flat_idx += 1;
        }
    }

    let mean = global_sum / global_count as f64;
    let rms = (global_sumsq / global_count as f64).sqrt();

    eprintln!("[glm-debug] {name}[0][:8]={first8:?}");
    eprintln!("[glm-debug] {name}[0] sum={row_sum:.6} l2={row_l2:.6}");
    eprintln!(
        "[glm-debug] {name} global min={global_min:.6} max={global_max:.6} mean={mean:.6} rms={rms:.6} hash=0x{hash:016x}"
    );
    eprintln!("[glm-debug] {name} probes={probes:?}");

    if let (Some(dir), Some(values)) = (dump_dir.as_ref(), flat_dump.as_ref()) {
        std::fs::create_dir_all(dir)?;
        let tensor_path = dir.join(format!("{name}.f32le"));
        let shape_path = dir.join(format!("{name}.shape.txt"));

        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for &v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&tensor_path, bytes)?;
        std::fs::write(&shape_path, format!("{rows} {cols}\n"))?;
        eprintln!(
            "[glm-debug] {name} dump={} shape={}",
            tensor_path.display(),
            shape_path.display()
        );
    }
    Ok(())
}

impl OcrEngine for GlmOcrModel {
    fn kind(&self) -> ModelKind {
        ModelKind::GlmOcr
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
        vision: VisionSettings,
        params: &DecodeParameters,
        stream: Option<&dyn Fn(usize, &[i64])>,
    ) -> Result<DecodeOutcome> {
        let total_timer = Timer::new("decode.generate");
        self.validate_decode_params(params)?;
        ensure!(
            params.use_cache,
            "GLM backend currently requires use_cache=true"
        );

        let prepared = self.prepare_inputs(tokenizer, prompt, images, vision)?;
        if params.max_new_tokens == 0 {
            total_timer.finish(|event| {
                event.add_field("prompt_tokens", prepared.prompt_len as u64);
                event.add_field("generated_tokens", 0u64);
                event.add_field("max_new_tokens", 0u64);
            });
            return Ok(DecodeOutcome {
                text: String::new(),
                prompt_tokens: prepared.prompt_len,
                response_tokens: 0,
                generated_tokens: Vec::new(),
            });
        }

        let eos_ids = self.effective_eos_ids();
        let mut cache = self.decoder.new_cache();
        let mut guard = self.decoder.prompt_guard(&mut cache);

        let prefill_timer = Timer::new("decode.prefill");
        let prefill = self.decoder.forward(
            None,
            Some(&prepared.embeddings),
            Some(&prepared.attention_mask),
            Some(&prepared.position_ids),
            Some(guard.cache()),
            params.use_cache,
        )?;
        prefill_timer.finish(|event| {
            event.add_field("prompt_tokens", prepared.prompt_len as u64);
            event.add_field("use_cache", params.use_cache);
        });

        let logits = prefill
            .logits
            .get(0)?
            .get(prepared.prompt_len.saturating_sub(1))?;

        #[cfg(feature = "trace-logits")]
        {
            if std::env::var_os("GLM_DEBUG_PREFILL").is_some() {
                let mut scored = logits
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .enumerate()
                    .collect::<Vec<_>>();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                eprintln!("[glm-debug] prefill top10 logits:");
                for (idx, score) in scored.into_iter().take(10) {
                    eprintln!("  id={idx} logit={score}");
                }
            }
        }

        let mut context_tokens = prepared.prompt_tokens;
        let mut next_position_base = prepared.next_position_base;
        let mut rng = init_rng(params.seed);
        let mut generated = Vec::with_capacity(params.max_new_tokens);
        let mut sampling_params = params.clone();
        sampling_params.no_repeat_ngram_size = None;
        let mut current = select_token_id(&logits, &sampling_params, &context_tokens, &mut rng)?;
        let decode_timer = Timer::new("decode.iterative");

        if eos_ids.contains(&current) {
            decode_timer.finish(|event| {
                event.add_field("steps", 0u64);
                event.add_field("max_new_tokens", params.max_new_tokens as u64);
            });
            total_timer.finish(|event| {
                event.add_field("prompt_tokens", context_tokens.len() as u64);
                event.add_field("generated_tokens", 0u64);
                event.add_field("max_new_tokens", params.max_new_tokens as u64);
            });
            return Ok(DecodeOutcome {
                text: String::new(),
                prompt_tokens: context_tokens.len(),
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
            if eos_ids.contains(&current) {
                break;
            }
            if generated.len() >= params.max_new_tokens {
                break;
            }

            let decode_ids = Tensor::from_vec(vec![current], (1, 1), &self.device)?;
            let pos = Tensor::from_vec(
                vec![next_position_base, next_position_base, next_position_base],
                (3, 1, 1),
                &self.device,
            )?;
            next_position_base += 1;

            let decode = self.decoder.forward(
                Some(&decode_ids),
                None,
                None,
                Some(&pos),
                Some(guard.cache()),
                params.use_cache,
            )?;
            let next_logits = decode.logits.get(0)?.get(0)?;

            #[cfg(feature = "trace-logits")]
            {
                if std::env::var_os("GLM_DEBUG_PREFILL").is_some() {
                    let mut scored = next_logits
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?
                        .into_iter()
                        .enumerate()
                        .collect::<Vec<_>>();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                    eprintln!("[glm-debug] decode step={} top10 logits:", generated.len());
                    for (idx, score) in scored.into_iter().take(10) {
                        eprintln!("  id={idx} logit={score}");
                    }
                }
            }

            current = select_token_id(&next_logits, &sampling_params, &context_tokens, &mut rng)?;
        }

        decode_timer.finish(|event| {
            event.add_field("steps", generated.len() as u64);
            event.add_field("max_new_tokens", params.max_new_tokens as u64);
        });

        let decoded = tokenizer
            .decode(
                &generated
                    .iter()
                    .filter_map(|&id| u32::try_from(id).ok())
                    .collect::<Vec<_>>(),
                true,
            )
            .unwrap_or_default();

        total_timer.finish(|event| {
            event.add_field("prompt_tokens", prepared.prompt_len as u64);
            event.add_field("generated_tokens", generated.len() as u64);
            event.add_field("max_new_tokens", params.max_new_tokens as u64);
        });

        Ok(DecodeOutcome {
            text: normalize_text(&decoded),
            prompt_tokens: prepared.prompt_len,
            response_tokens: generated.len(),
            generated_tokens: generated,
        })
    }
}

pub fn load_model(args: ModelLoadArgs<'_>) -> Result<Box<dyn OcrEngine>> {
    if args.kind != ModelKind::GlmOcr {
        return Err(anyhow!("unsupported model kind: {:?}", args.kind));
    }
    Ok(Box::new(GlmOcrModel::load(&args)?))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenType {
    Image,
    Video,
    Text,
}

struct PreparedInputs {
    embeddings: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    prompt_tokens: Vec<i64>,
    next_position_base: i64,
    prompt_len: usize,
}
