use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor, shape::D};
use candle_nn::VarBuilder;
use image::GenericImageView;
use image::{DynamicImage, Rgb, RgbImage, imageops};

use crate::{
    config::{DeepseekOcrConfig, ProjectorConfig, load_ocr_config},
    transformer::{
        cache::{DynamicCache, PromptCacheGuard},
        model::{DeepseekLanguageModel, LanguageModelOutput},
    },
    vision::{
        ClipDebugTrace, ClipVisionModel, SamBackbone, SamDebugTrace, dynamic_preprocess,
        resample::resize_bicubic,
    },
};

pub const DEFAULT_WEIGHTS_PATH: &str = "DeepSeek-OCR/model-00001-of-000001.safetensors";

/// Vision inputs associated with a single batch element.
#[derive(Clone, Copy)]
pub struct VisionInput<'a> {
    pub global: &'a Tensor,
    pub patches: Option<&'a Tensor>,
    pub crop_shape: Option<(usize, usize)>,
}

/// Owned buffers backing a [`VisionInput`].
pub struct OwnedVisionInput {
    pub global: Tensor,
    pub patches: Option<Tensor>,
    pub crop_shape: Option<(usize, usize)>,
}

impl OwnedVisionInput {
    pub fn as_ref(&self) -> VisionInput<'_> {
        VisionInput {
            global: &self.global,
            patches: self.patches.as_ref(),
            crop_shape: self.crop_shape,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VisionProjectionOutputs {
    pub global_pre: Tensor,
    pub local_pre: Option<Tensor>,
    pub global_post: Tensor,
    pub local_post: Option<Tensor>,
    pub global_tokens: Tensor,
    pub local_tokens: Option<Tensor>,
    pub fused_tokens: Tensor,
}

struct VisionProcessArtifacts {
    fused_tokens: Tensor,
    global_pre: Tensor,
    local_pre: Option<Tensor>,
    global_post: Tensor,
    local_post: Option<Tensor>,
    global_tokens: Tensor,
    local_tokens: Option<Tensor>,
}

#[cfg_attr(not(test), allow(dead_code))]
pub struct VisionDebugFeatures {
    pub global_clip: Tensor,
    pub global_sam: Tensor,
    pub local_clip: Option<Tensor>,
    pub local_sam: Option<Tensor>,
    pub global_clip_trace: ClipDebugTrace,
    pub local_clip_trace: Option<ClipDebugTrace>,
    pub global_sam_trace: SamDebugTrace,
    pub local_sam_trace: Option<SamDebugTrace>,
}

/// Options controlling autoregressive generation.
pub struct GenerateOptions<'a> {
    pub attention_mask: Option<&'a Tensor>,
    pub position_ids: Option<&'a Tensor>,
    pub images_seq_mask: Option<&'a Tensor>,
    pub image_inputs: Option<&'a [Option<VisionInput<'a>>]>,
    pub image_embeddings: Option<&'a [Tensor]>,
    pub max_new_tokens: usize,
    pub eos_token_id: Option<i64>,
    pub progress_callback: Option<&'a dyn Fn(usize, &[i64])>,
    pub use_cache: bool,
}

impl<'a> GenerateOptions<'a> {
    pub fn new(max_new_tokens: usize) -> Self {
        Self {
            attention_mask: None,
            position_ids: None,
            images_seq_mask: None,
            image_inputs: None,
            image_embeddings: None,
            max_new_tokens,
            eos_token_id: None,
            progress_callback: None,
            use_cache: true,
        }
    }
}

struct ImageProjector {
    input_dim: usize,
    hidden: usize,
    weight: Tensor,
    bias: Option<Tensor>,
    image_newline: Tensor,
    view_separator: Tensor,
}

impl ImageProjector {
    fn load(vb: &VarBuilder, cfg: &ProjectorConfig) -> Result<Self> {
        let input_dim = cfg
            .input_dim
            .with_context(|| "projector input_dim missing from config")?;
        ensure!(
            cfg.projector_type == "linear",
            "unsupported projector_type `{}`",
            cfg.projector_type
        );

        let model_vb = vb.pp("model");
        let projector_vb = model_vb.pp("projector");
        let layers_vb = projector_vb.pp("layers");

        let weight = layers_vb
            .get((cfg.n_embed, input_dim), "weight")
            .with_context(|| "missing projector weight tensor")?
            .contiguous()?;
        let bias = if layers_vb.contains_tensor("bias") {
            Some(
                layers_vb
                    .get(cfg.n_embed, "bias")
                    .with_context(|| "missing projector bias tensor")?
                    .contiguous()?,
            )
        } else {
            None
        };
        let image_newline = model_vb
            .get(cfg.n_embed, "image_newline")
            .with_context(|| "missing projector image_newline tensor")?
            .contiguous()?;
        let view_separator = model_vb
            .get(cfg.n_embed, "view_seperator")
            .with_context(|| "missing projector view_seperator tensor")?
            .contiguous()?;

        Ok(Self {
            input_dim,
            hidden: cfg.n_embed,
            weight,
            bias,
            image_newline,
            view_separator,
        })
    }

    fn project(&self, input: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims();
        ensure!(
            !dims.is_empty(),
            "projector input must have rank >= 1, received {:?}",
            dims
        );
        let last_dim = *dims.last().expect("at least one dim");
        ensure!(
            last_dim == self.input_dim,
            "projector expected input dim {}, got {}",
            self.input_dim,
            last_dim
        );
        let leading = dims[..dims.len() - 1].iter().product::<usize>();
        let flat = input.reshape((leading, self.input_dim))?;
        let weight_t = self.weight.transpose(0, 1)?;
        let mut proj = flat.matmul(&weight_t)?;
        if let Some(bias) = &self.bias {
            proj = proj.broadcast_add(&bias.reshape((1, self.hidden))?)?;
        }
        proj.reshape(
            dims[..dims.len() - 1]
                .iter()
                .copied()
                .chain(std::iter::once(self.hidden))
                .collect::<Vec<_>>(),
        )
        .context("failed to reshape projector output")
    }

    fn adapt_tokens(&self, tensor: &Tensor, dtype: DType, device: &Device) -> Result<Tensor> {
        let tensor = tensor.to_device(device)?;
        if tensor.dtype() == dtype {
            Ok(tensor)
        } else {
            tensor
                .to_dtype(dtype)
                .context("failed to cast image embeddings")
        }
    }

    fn placeholders(&self, count: usize, dtype: DType, device: &Device) -> Result<Tensor> {
        if count == 0 {
            return Ok(Tensor::zeros((0, self.hidden), dtype, device)?);
        }
        let newline = self.adapt_tokens(&self.image_newline, dtype, device)?;
        let mut tokens = newline
            .unsqueeze(0)?
            .expand((count, self.hidden))?
            .contiguous()?;
        let separator = self
            .adapt_tokens(&self.view_separator, dtype, device)?
            .unsqueeze(0)?;
        tokens = tokens.slice_assign(&[count - 1..count, 0..self.hidden], &separator)?;
        Ok(tokens)
    }

    fn image_newline_token(&self, dtype: DType, device: &Device) -> Result<Tensor> {
        self.adapt_tokens(&self.image_newline, dtype, device)
    }

    fn view_separator_token(&self, dtype: DType, device: &Device) -> Result<Tensor> {
        self.adapt_tokens(&self.view_separator, dtype, device)
    }

    fn hidden_size(&self) -> usize {
        self.hidden
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }
}

/// High-level multimodal container that will eventually wrap the vision towers, projector, and
/// language model. For now it wires the language stack so we can exercise text-only inference.
pub struct DeepseekOcrModel {
    cfg: Arc<DeepseekOcrConfig>,
    language: DeepseekLanguageModel,
    projector_cfg: Arc<ProjectorConfig>,
    projector: ImageProjector,
    vision: VisionModules,
    device: Device,
    dtype: DType,
    weights_path: PathBuf,
}

struct VisionModules {
    sam: SamBackbone,
    clip: ClipVisionModel,
}

impl DeepseekOcrModel {
    /// Load the OCR model from disk, pulling configuration and language-model weights.
    ///
    /// The vision/projector paths are stubbed for now; they will be filled in once the Candle
    /// kernels land. `device` controls where tensors are allocated (CPU/GPU).
    pub fn load(
        config_path: Option<&Path>,
        weights_path: Option<&Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let cfg = Arc::new(load_ocr_config(config_path)?);
        let language_cfg = Arc::new(cfg.resolved_language_config()?);
        let resolved_weights = weights_path
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_WEIGHTS_PATH));
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[resolved_weights.as_path()], dtype, &device)
        }
        .with_context(|| format!("failed to mmap weights at {}", resolved_weights.display()))?;
        let language = DeepseekLanguageModel::load(language_cfg, &vb)
            .context("failed to load language model")?;
        let projector_cfg = Arc::new(
            cfg.resolved_projector_config()
                .context("projector configuration missing")?,
        );
        ensure!(
            projector_cfg.n_embed == language.config().hidden_size,
            "projector n_embed {} mismatches language hidden size {}",
            projector_cfg.n_embed,
            language.config().hidden_size
        );
        let projector = ImageProjector::load(&vb, projector_cfg.as_ref())
            .context("failed to load image projector")?;
        let sam = SamBackbone::new(cfg.as_ref(), &vb.pp("model").pp("sam_model"))
            .context("failed to load SAM backbone")?;
        let clip = ClipVisionModel::load(cfg.as_ref(), &vb.pp("model").pp("vision_model"))
            .context("failed to load CLIP vision model")?;
        let vision = VisionModules { sam, clip };

        Ok(Self {
            cfg,
            language,
            projector_cfg,
            projector,
            vision,
            device,
            dtype,
            weights_path: resolved_weights,
        })
    }

    /// Access the currently loaded configuration.
    pub fn config(&self) -> &DeepseekOcrConfig {
        self.cfg.as_ref()
    }

    /// Device backing the allocated tensors.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// DType the model was loaded with.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Path the weights were loaded from (useful for logging).
    pub fn weights_path(&self) -> &Path {
        &self.weights_path
    }

    /// Borrow the language-only component.
    pub fn language_model(&self) -> &DeepseekLanguageModel {
        &self.language
    }

    /// Whether flash attention is enabled for the underlying decoder.
    pub fn flash_attention_enabled(&self) -> bool {
        self.language.flash_attention_enabled()
    }

    /// Access the projector configuration.
    pub fn projector_config(&self) -> &ProjectorConfig {
        self.projector_cfg.as_ref()
    }

    /// Construct a fresh dynamic cache sized for this model.
    pub fn new_cache(&self) -> DynamicCache {
        let layers = self.language.transformer_weights().layers.len();
        DynamicCache::with_num_layers(layers)
    }

    /// Helper to guard prompt-scoped cache state.
    pub fn prompt_guard<'a>(&'a self, cache: &'a mut DynamicCache) -> PromptCacheGuard<'a> {
        self.language.prompt_guard(cache)
    }

    #[doc(hidden)]
    pub fn sam_backbone(&self) -> &SamBackbone {
        &self.vision.sam
    }

    /// Forward pass through the multimodal stack, applying optional image-token injection.
    pub fn forward<'a>(
        &self,
        input_ids: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        images_seq_mask: Option<&Tensor>,
        vision_inputs: Option<&'a [Option<VisionInput<'a>>]>,
        image_embeddings: Option<&'a [Tensor]>,
        cache: Option<&mut DynamicCache>,
        use_cache: bool,
    ) -> Result<LanguageModelOutput> {
        ensure!(
            input_ids.is_some() ^ inputs_embeds.is_some(),
            "provide exactly one of input_ids or inputs_embeds"
        );
        ensure!(
            !use_cache || cache.is_some(),
            "use_cache=true requires a mutable DynamicCache"
        );
        if vision_inputs.is_some() || image_embeddings.is_some() {
            ensure!(
                images_seq_mask.is_some(),
                "image masks required when providing image inputs or embeddings"
            );
        }

        let mut embeddings = match inputs_embeds {
            Some(t) => t.clone(),
            None => {
                let ids = input_ids.expect("input_ids validity checked above");
                self.language.embed_tokens(ids)?
            }
        };

        let computed_embeddings = if image_embeddings.is_none() {
            if let Some(inputs) = vision_inputs {
                Some(self.compute_image_embeddings(inputs)?)
            } else {
                None
            }
        } else {
            None
        };
        let image_embeddings_slice = image_embeddings
            .map(Some)
            .unwrap_or_else(|| computed_embeddings.as_ref().map(|v| v.as_slice()));

        if let Some(mask) = images_seq_mask {
            embeddings = self.inject_image_tokens(embeddings, mask, image_embeddings_slice)?;
        }

        self.language.forward(
            None,
            Some(&embeddings),
            attention_mask,
            position_ids,
            cache,
            use_cache,
        )
    }

    /// Convenience wrapper around the language-model forward path without image tokens.
    pub fn forward_language(
        &self,
        input_ids: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        cache: Option<&mut DynamicCache>,
        use_cache: bool,
    ) -> Result<LanguageModelOutput> {
        self.forward(
            input_ids,
            inputs_embeds,
            attention_mask,
            position_ids,
            None,
            None,
            None,
            cache,
            use_cache,
        )
    }

    pub fn compute_image_embeddings(
        &self,
        inputs: &[Option<VisionInput<'_>>],
    ) -> Result<Vec<Tensor>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            if let Some(vision_input) = input {
                outputs.push(self.process_vision_input(vision_input)?);
            } else {
                outputs.push(Tensor::zeros(
                    (0, self.projector.hidden_size()),
                    self.dtype,
                    self.device(),
                )?);
            }
        }
        Ok(outputs)
    }

    pub fn compute_vision_projection(
        &self,
        input: &VisionInput<'_>,
    ) -> Result<VisionProjectionOutputs> {
        let artifacts = self.process_vision_input_full(input)?;
        let VisionProcessArtifacts {
            fused_tokens,
            global_pre,
            local_pre,
            global_post,
            local_post,
            global_tokens,
            local_tokens,
        } = artifacts;

        let (batch, _, _) = global_pre
            .shape()
            .dims3()
            .context("global pre tokens must be 3D")?;
        ensure!(
            batch == 1,
            "global pre tokens expect batch size 1, got {batch}"
        );
        let global_pre_flat = global_pre
            .get(0)?
            .contiguous()
            .context("global pre flat not contiguous")?;

        let (post_batch, _, _) = global_post
            .shape()
            .dims3()
            .context("global post tokens must be 3D")?;
        ensure!(
            post_batch == 1,
            "global post tokens expect batch size 1, got {post_batch}"
        );
        let global_post_flat = global_post
            .get(0)?
            .contiguous()
            .context("global post flat not contiguous")?;

        let local_pre_flat = if let Some(local_pre) = local_pre {
            let (patches, seq, hidden) = local_pre
                .shape()
                .dims3()
                .context("local pre tokens must be 3D")?;
            Some(
                local_pre
                    .reshape((patches * seq, hidden))?
                    .contiguous()
                    .context("local pre flat not contiguous")?,
            )
        } else {
            None
        };

        let local_post_flat = if let Some(local_post) = local_post {
            let (patches, seq, hidden) = local_post
                .shape()
                .dims3()
                .context("local post tokens must be 3D")?;
            Some(
                local_post
                    .reshape((patches * seq, hidden))?
                    .contiguous()
                    .context("local post flat not contiguous")?,
            )
        } else {
            None
        };

        Ok(VisionProjectionOutputs {
            global_pre: global_pre_flat,
            local_pre: local_pre_flat,
            global_post: global_post_flat,
            local_post: local_post_flat,
            global_tokens,
            local_tokens,
            fused_tokens,
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn compute_vision_debug_features(
        &self,
        input: &VisionInput<'_>,
    ) -> Result<VisionDebugFeatures> {
        let global = self
            .prepare_image_tensor(input.global)
            .context("invalid global image tensor")?;
        let (sam_global_raw, sam_trace_global_raw) = self
            .vision
            .sam
            .forward_with_trace(&global)
            .context("sam forward (global)")?;
        let sam_global = sam_global_raw
            .contiguous()
            .context("sam global not contiguous")?;
        let clip_trace_global_raw = self
            .vision
            .clip
            .forward_with_trace(&global, Some(&sam_global))
            .context("clip forward (global)")?;
        let clip_global = clip_trace_global_raw
            .output
            .contiguous()
            .context("clip global output not contiguous")?;

        let clip_global_layers = clip_trace_global_raw
            .layer_outputs
            .into_iter()
            .enumerate()
            .map(|(idx, tensor)| {
                tensor
                    .contiguous()
                    .with_context(|| format!("clip global layer {idx} not contiguous"))
            })
            .collect::<Result<Vec<_>>>()?;
        let clip_trace_global = ClipDebugTrace {
            embeddings: clip_trace_global_raw
                .embeddings
                .contiguous()
                .context("clip global embeddings not contiguous")?,
            pre_layernorm: clip_trace_global_raw
                .pre_layernorm
                .contiguous()
                .context("clip global pre_layernorm not contiguous")?,
            layer_outputs: clip_global_layers,
            output: clip_global.clone(),
        };

        let sam_trace_global = SamDebugTrace {
            patch_embed: sam_trace_global_raw
                .patch_embed
                .contiguous()
                .context("sam global patch_embed not contiguous")?,
            pos_added: match sam_trace_global_raw.pos_added {
                Some(tensor) => Some(
                    tensor
                        .contiguous()
                        .context("sam global pos_added not contiguous")?,
                ),
                None => None,
            },
            block_outputs: sam_trace_global_raw
                .block_outputs
                .into_iter()
                .enumerate()
                .map(|(idx, tensor)| {
                    tensor
                        .contiguous()
                        .with_context(|| format!("sam global block {idx} not contiguous"))
                })
                .collect::<Result<Vec<_>>>()?,
            neck_conv1: sam_trace_global_raw
                .neck_conv1
                .contiguous()
                .context("sam global neck conv1 not contiguous")?,
            neck_norm1: sam_trace_global_raw
                .neck_norm1
                .contiguous()
                .context("sam global neck norm1 not contiguous")?,
            neck_conv2: sam_trace_global_raw
                .neck_conv2
                .contiguous()
                .context("sam global neck conv2 not contiguous")?,
            neck_norm2: sam_trace_global_raw
                .neck_norm2
                .contiguous()
                .context("sam global neck norm2 not contiguous")?,
            net2: sam_trace_global_raw
                .net2
                .contiguous()
                .context("sam global net2 not contiguous")?,
            net3: sam_trace_global_raw
                .net3
                .contiguous()
                .context("sam global net3 not contiguous")?,
        };

        let mut local_clip_opt = None;
        let mut local_sam_opt = None;
        let mut local_clip_trace_opt = None;
        let mut local_sam_trace_opt = None;

        if let Some(patches) = input.patches {
            let patches = self
                .prepare_image_tensor(patches)
                .context("invalid patch tensor")?;
            let (patch_batch, _c, _h, _w) =
                patches.shape().dims4().context("patch tensor must be 4D")?;
            if patch_batch > 0 {
                let (sam_local_raw, sam_trace_local_raw) = self
                    .vision
                    .sam
                    .forward_with_trace(&patches)
                    .context("sam forward (local)")?;
                let sam_local = sam_local_raw
                    .contiguous()
                    .context("sam local not contiguous")?;
                let clip_trace_local_raw = self
                    .vision
                    .clip
                    .forward_with_trace(&patches, Some(&sam_local))
                    .context("clip forward (local)")?;
                let clip_local = clip_trace_local_raw
                    .output
                    .contiguous()
                    .context("clip local output not contiguous")?;

                let clip_local_layers = clip_trace_local_raw
                    .layer_outputs
                    .into_iter()
                    .enumerate()
                    .map(|(idx, tensor)| {
                        tensor
                            .contiguous()
                            .with_context(|| format!("clip local layer {idx} not contiguous"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let clip_trace_local = ClipDebugTrace {
                    embeddings: clip_trace_local_raw
                        .embeddings
                        .contiguous()
                        .context("clip local embeddings not contiguous")?,
                    pre_layernorm: clip_trace_local_raw
                        .pre_layernorm
                        .contiguous()
                        .context("clip local pre_layernorm not contiguous")?,
                    layer_outputs: clip_local_layers,
                    output: clip_local.clone(),
                };

                let sam_trace_local = SamDebugTrace {
                    patch_embed: sam_trace_local_raw
                        .patch_embed
                        .contiguous()
                        .context("sam local patch_embed not contiguous")?,
                    pos_added: match sam_trace_local_raw.pos_added {
                        Some(tensor) => Some(
                            tensor
                                .contiguous()
                                .context("sam local pos_added not contiguous")?,
                        ),
                        None => None,
                    },
                    block_outputs: sam_trace_local_raw
                        .block_outputs
                        .into_iter()
                        .enumerate()
                        .map(|(idx, tensor)| {
                            tensor
                                .contiguous()
                                .with_context(|| format!("sam local block {idx} not contiguous"))
                        })
                        .collect::<Result<Vec<_>>>()?,
                    neck_conv1: sam_trace_local_raw
                        .neck_conv1
                        .contiguous()
                        .context("sam local neck conv1 not contiguous")?,
                    neck_norm1: sam_trace_local_raw
                        .neck_norm1
                        .contiguous()
                        .context("sam local neck norm1 not contiguous")?,
                    neck_conv2: sam_trace_local_raw
                        .neck_conv2
                        .contiguous()
                        .context("sam local neck conv2 not contiguous")?,
                    neck_norm2: sam_trace_local_raw
                        .neck_norm2
                        .contiguous()
                        .context("sam local neck norm2 not contiguous")?,
                    net2: sam_trace_local_raw
                        .net2
                        .contiguous()
                        .context("sam local net2 not contiguous")?,
                    net3: sam_trace_local_raw
                        .net3
                        .contiguous()
                        .context("sam local net3 not contiguous")?,
                };

                local_clip_opt = Some(clip_local);
                local_sam_opt = Some(sam_local);
                local_clip_trace_opt = Some(clip_trace_local);
                local_sam_trace_opt = Some(sam_trace_local);
            }
        }

        Ok(VisionDebugFeatures {
            global_clip: clip_global,
            global_sam: sam_global,
            local_clip: local_clip_opt,
            local_sam: local_sam_opt,
            global_clip_trace: clip_trace_global,
            local_clip_trace: local_clip_trace_opt,
            global_sam_trace: sam_trace_global,
            local_sam_trace: local_sam_trace_opt,
        })
    }

    fn process_vision_input_full(&self, input: &VisionInput<'_>) -> Result<VisionProcessArtifacts> {
        let newline = self
            .projector
            .image_newline_token(self.dtype, self.device())
            .context("failed to adapt image_newline token")?;

        let global = self
            .prepare_image_tensor(input.global)
            .context("invalid global image tensor")?;
        let sam_global = self
            .vision
            .sam
            .forward(&global)
            .context("sam forward (global)")?;
        let clip_global = self
            .vision
            .clip
            .forward(&global, Some(&sam_global))
            .context("clip forward (global)")?;
        let global_pre = self
            .build_clip_sam_tokens(&clip_global, &sam_global)
            .context("concat global clip+sam tokens")?
            .contiguous()
            .context("global pre tokens not contiguous")?;
        let global_post = self
            .projector
            .project(&global_pre)
            .context("project global features")?
            .contiguous()
            .context("global post tokens not contiguous")?;
        let global_tokens = self
            .format_global_tokens(&global_post, &newline)
            .context("format global tokens")?
            .contiguous()
            .context("global tokens not contiguous")?;

        let mut local_pre_opt = None;
        let mut local_post_opt = None;
        let mut local_tokens_opt = None;

        if let Some(patches) = input.patches {
            let crop_shape = input
                .crop_shape
                .context("crop_shape required when patches are provided")?;
            let patches = self
                .prepare_image_tensor(patches)
                .context("invalid patch tensor")?;
            let (patch_batch, _c, _h, _w) = patches
                .shape()
                .dims4()
                .context("patch tensor must be 4D (batch, channels, height, width)")?;
            if patch_batch > 0 {
                let sam_local = self
                    .vision
                    .sam
                    .forward(&patches)
                    .context("sam forward (local)")?;
                let clip_local = self
                    .vision
                    .clip
                    .forward(&patches, Some(&sam_local))
                    .context("clip forward (local)")?;
                let local_pre = self
                    .build_clip_sam_tokens(&clip_local, &sam_local)
                    .context("concat local clip+sam tokens")?
                    .contiguous()
                    .context("local pre tokens not contiguous")?;
                let local_post = self
                    .projector
                    .project(&local_pre)
                    .context("project local features")?
                    .contiguous()
                    .context("local post tokens not contiguous")?;
                let local_tokens = self
                    .format_local_tokens(&local_post, crop_shape, &newline)
                    .context("format local tokens")?
                    .contiguous()
                    .context("local tokens not contiguous")?;
                local_pre_opt = Some(local_pre);
                local_post_opt = Some(local_post);
                local_tokens_opt = Some(local_tokens);
            }
        }

        let mut segments = Vec::new();
        if let Some(local_tokens) = local_tokens_opt.clone() {
            segments.push(local_tokens);
        }
        segments.push(global_tokens.clone());
        let view_separator = self
            .projector
            .view_separator_token(self.dtype, self.device())
            .context("failed to adapt view separator token")?
            .reshape((1, self.projector.hidden_size()))?
            .contiguous()
            .context("view separator not contiguous")?;
        segments.push(view_separator);
        let fused_tokens = Tensor::cat(&segments, 0)
            .context("failed to concatenate image segments")?
            .contiguous()
            .context("fused tokens not contiguous")?;

        Ok(VisionProcessArtifacts {
            fused_tokens,
            global_pre,
            local_pre: local_pre_opt,
            global_post,
            local_post: local_post_opt,
            global_tokens,
            local_tokens: local_tokens_opt,
        })
    }

    fn process_vision_input(&self, input: &VisionInput<'_>) -> Result<Tensor> {
        Ok(self.process_vision_input_full(input)?.fused_tokens)
    }

    fn prepare_image_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut image = if tensor.rank() == 3 {
            tensor.unsqueeze(0)?
        } else {
            tensor.clone()
        };
        ensure!(
            image.rank() == 4,
            "image tensor must have rank 4 (batch, channels, height, width)"
        );
        if !image.device().same_device(self.device()) {
            image = image.to_device(self.device())?;
        }
        if image.dtype() != self.dtype {
            image = image.to_dtype(self.dtype)?;
        }
        Ok(image.contiguous()?)
    }

    fn build_clip_sam_tokens(&self, clip: &Tensor, sam: &Tensor) -> Result<Tensor> {
        let (batch, clip_seq, clip_hidden) = clip
            .shape()
            .dims3()
            .context("clip output must be [batch, seq, hidden]")?;
        ensure!(clip_seq > 0, "clip output missing sequence dimension");
        let clip_tokens = clip
            .narrow(D::Minus2, 1, clip_seq - 1)?
            .contiguous()
            .context("clip token slice not contiguous")?;
        let (sam_batch, sam_channels, sam_h, sam_w) = sam
            .shape()
            .dims4()
            .context("sam output must be [batch, channels, height, width]")?;
        ensure!(
            sam_batch == batch,
            "sam batch {} does not match clip batch {}",
            sam_batch,
            batch
        );
        let sam_tokens = sam
            .reshape((batch, sam_channels, sam_h * sam_w))?
            .transpose(1, 2)?
            .contiguous()
            .context("sam token transpose not contiguous")?;
        let (_, sam_seq, sam_hidden) = sam_tokens
            .shape()
            .dims3()
            .context("sam tokens reshape failed")?;
        let (_, clip_seq_trimmed, _) = clip_tokens
            .shape()
            .dims3()
            .context("clip token slice should be 3D")?;
        ensure!(
            clip_seq_trimmed == sam_seq,
            "clip tokens ({clip_seq_trimmed}) do not match sam tokens ({sam_seq})"
        );
        ensure!(
            clip_hidden + sam_hidden == self.projector.input_dim(),
            "combined hidden dims {}+{} do not match projector input {}",
            clip_hidden,
            sam_hidden,
            self.projector.input_dim()
        );
        let combined = Tensor::cat(&[clip_tokens, sam_tokens], D::Minus1)?;
        Ok(combined)
    }

    fn format_global_tokens(&self, projected: &Tensor, newline: &Tensor) -> Result<Tensor> {
        let (batch, seq, hidden) = projected
            .shape()
            .dims3()
            .context("projected global tokens must be 3D")?;
        ensure!(batch == 1, "global view expects batch size 1, got {batch}");
        let side = (seq as f64).sqrt() as usize;
        ensure!(
            side * side == seq,
            "global token count {} is not a perfect square",
            seq
        );
        let grid = projected
            .get(0)?
            .reshape((side, side, hidden))?
            .contiguous()
            .context("global grid reshape not contiguous")?;
        self.append_row_breaks(grid, newline)
    }

    fn format_local_tokens(
        &self,
        projected: &Tensor,
        crop_shape: (usize, usize),
        newline: &Tensor,
    ) -> Result<Tensor> {
        let (patches, seq, hidden) = projected
            .shape()
            .dims3()
            .context("projected local tokens must be 3D")?;
        let (width_crops, height_crops) = crop_shape;
        ensure!(
            patches == width_crops * height_crops,
            "patch count {} does not match crop grid {}x{}",
            patches,
            width_crops,
            height_crops
        );
        let side = (seq as f64).sqrt() as usize;
        ensure!(
            side * side == seq,
            "local token count {} is not a perfect square",
            seq
        );
        let grid = projected
            .reshape((height_crops, width_crops, side, side, hidden))?
            .permute((0, 2, 1, 3, 4))?
            .reshape((height_crops * side, width_crops * side, hidden))?
            .contiguous()
            .context("local grid reshape not contiguous")?;
        self.append_row_breaks(grid, newline)
    }

    fn append_row_breaks(&self, grid: Tensor, newline: &Tensor) -> Result<Tensor> {
        let (rows, cols, hidden) = grid
            .shape()
            .dims3()
            .context("grid must be [rows, cols, hidden]")?;
        let grid3 = grid.reshape((rows, cols, hidden))?;
        let newline = newline
            .reshape((1, 1, hidden))?
            .expand((rows, 1, hidden))?
            .contiguous()?;
        let with_breaks = Tensor::cat(&[grid3, newline], 1)?;
        Ok(with_breaks.reshape((rows * (cols + 1), hidden))?)
    }

    /// Construct normalized tensors for a single multimodal example.
    pub fn prepare_vision_input_from_image(
        &self,
        image: &DynamicImage,
        base_size: u32,
        image_size: u32,
        crop_mode: bool,
    ) -> Result<OwnedVisionInput> {
        let global_view = build_global_view(image, base_size);
        let global = image_to_tensor(&global_view, self.device(), self.dtype)?
            .unsqueeze(0)?
            .contiguous()?;

        let (patches, crop_shape) = if crop_mode {
            let preprocess = dynamic_preprocess(image, 2, 9, image_size, false);
            let crop = (preprocess.ratio.0 as usize, preprocess.ratio.1 as usize);
            let tiles = preprocess.tiles;
            if tiles.is_empty() {
                (None, Some(crop))
            } else {
                let mut tensors = Vec::with_capacity(tiles.len());
                for tile in tiles {
                    tensors.push(image_to_tensor(&tile, self.device(), self.dtype)?);
                }
                let stacked = Tensor::stack(&tensors, 0)?.contiguous()?;
                (Some(stacked), Some(crop))
            }
        } else {
            (None, None)
        };

        Ok(OwnedVisionInput {
            global,
            patches,
            crop_shape,
        })
    }

    fn inject_image_tokens(
        &self,
        embeddings: Tensor,
        mask: &Tensor,
        image_embeddings: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = embeddings.shape().dims3()?;
        if let Some(tokens) = image_embeddings {
            ensure!(
                tokens.len() == batch,
                "image_embeddings batch {} does not match embeddings batch {batch}",
                tokens.len()
            );
        }
        let mask = if mask.dtype() == DType::U8 {
            mask.clone()
        } else {
            mask.to_dtype(DType::U8)?
        };
        let (mask_batch, mask_seq) = mask
            .shape()
            .dims2()
            .context("images_seq_mask must have shape [batch, seq_len]")?;
        ensure!(
            mask_batch == batch && mask_seq == seq_len,
            "images_seq_mask shape ({mask_batch}, {mask_seq}) does not match embeddings ({batch}, {seq_len})"
        );

        let dtype = embeddings.dtype();
        let device = embeddings.device();
        let mut rows = Vec::with_capacity(batch);
        for b in 0..batch {
            let row = embeddings
                .get(b)?
                .reshape((seq_len, hidden))?
                .contiguous()?;
            let mask_row = mask.get(b)?.reshape((seq_len,))?;
            let mask_vec = mask_row.to_vec1::<u8>()?;
            let positions: Vec<usize> = mask_vec
                .iter()
                .enumerate()
                .filter_map(|(idx, &flag)| (flag != 0).then_some(idx))
                .collect();
            if positions.is_empty() {
                rows.push(row);
                continue;
            }
            let replacements = if let Some(tokens) = image_embeddings {
                let per_batch = tokens
                    .get(b)
                    .context("image_embeddings missing entry for batch row")?;
                let adapted = self
                    .projector
                    .adapt_tokens(per_batch, dtype, device)?
                    .contiguous()?;
                let (count, embed_dim) = adapted
                    .shape()
                    .dims2()
                    .context("image embeddings must have shape [tokens, hidden]")?;
                ensure!(
                    count == positions.len(),
                    "image embeddings provide {} tokens but mask requires {}",
                    count,
                    positions.len()
                );
                ensure!(
                    embed_dim == hidden,
                    "image embedding hidden dim {} does not match language hidden size {}",
                    embed_dim,
                    hidden
                );
                adapted
            } else {
                self.projector
                    .placeholders(positions.len(), dtype, device)?
            };
            let replacements = replacements.contiguous()?;

            let replacements_full = Tensor::zeros((seq_len, hidden), dtype, device)?;
            let positions_i64: Vec<i64> = positions.iter().map(|&p| p as i64).collect();
            let idx_tensor = Tensor::from_vec(positions_i64, (positions.len(),), device)?
                .to_dtype(DType::I64)?;
            let idx_matrix = idx_tensor
                .reshape((positions.len(), 1))?
                .expand((positions.len(), hidden))?
                .contiguous()?;
            replacements_full.scatter_add_set(&idx_matrix, &replacements, 0)?;

            let mask_float = mask_row
                .to_dtype(dtype)?
                .reshape((seq_len, 1))?
                .contiguous()?;
            let ones = Tensor::ones((seq_len, 1), dtype, device)?;
            let keep = ones.sub(&mask_float)?;
            let updated = row.broadcast_mul(&keep)?.add(&replacements_full)?;
            rows.push(updated);
        }
        Ok(Tensor::stack(&rows, 0)?)
    }

    #[doc(hidden)]
    pub fn inject_image_tokens_for_tests(
        &self,
        embeddings: Tensor,
        mask: &Tensor,
        image_embeddings: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        self.inject_image_tokens(embeddings, mask, image_embeddings)
    }

    /// Greedy autoregressive generation for the multimodal model.
    pub fn generate(&self, input_ids: &Tensor, options: GenerateOptions<'_>) -> Result<Tensor> {
        ensure!(
            input_ids.rank() == 2,
            "generate expects input_ids with shape [batch, seq]"
        );
        let (batch, seq_len) = input_ids.shape().dims2()?;
        ensure!(
            batch == 1,
            "generate currently supports batch size 1 (got {batch})"
        );
        if !options.use_cache {
            return self.generate_without_cache(input_ids, options);
        }
        let progress_callback = options.progress_callback;
        if options.max_new_tokens == 0 {
            return self.empty_generation();
        }

        let mut cache = self.new_cache();
        let mut guard = self.prompt_guard(&mut cache);
        let prefill = self.forward(
            Some(input_ids),
            None,
            options.attention_mask,
            options.position_ids,
            options.images_seq_mask,
            options.image_inputs,
            options.image_embeddings,
            Some(guard.cache()),
            true,
        )?;
        let logits = prefill
            .logits
            .get(0)
            .context("prefill logits missing batch dimension")?;
        let last_logits = logits
            .get(seq_len - 1)
            .context("prefill logits missing final timestep")?;
        let mut current = self.select_token_id(&last_logits)?;
        if let Some(eos) = options.eos_token_id {
            if current == eos {
                return self.empty_generation();
            }
        }

        let mut generated = Vec::with_capacity(options.max_new_tokens);
        for step in 0..options.max_new_tokens {
            generated.push(current);
            if let Some(cb) = progress_callback {
                cb(generated.len(), &generated);
            }
            if step + 1 == options.max_new_tokens {
                break;
            }
            let next_input =
                Tensor::from_vec(vec![current], (1, 1), self.device())?.to_dtype(DType::I64)?;
            let decode = self.forward(
                Some(&next_input),
                None,
                None,
                None,
                None,
                None,
                None,
                Some(guard.cache()),
                true,
            )?;
            let next_logits = decode
                .logits
                .get(0)
                .context("decode logits missing batch dimension")?
                .get(0)
                .context("decode logits missing timestep")?;
            current = self.select_token_id(&next_logits)?;
            if let Some(eos) = options.eos_token_id {
                if current == eos {
                    break;
                }
            }
        }
        let len = generated.len();
        Ok(Tensor::from_vec(generated, (1, len), self.device())?.to_dtype(DType::I64)?)
    }

    fn generate_without_cache(
        &self,
        input_ids: &Tensor,
        options: GenerateOptions<'_>,
    ) -> Result<Tensor> {
        ensure!(
            input_ids.rank() == 2,
            "generate expects input_ids with shape [batch, seq]"
        );
        let (batch, seq_len) = input_ids.shape().dims2()?;
        ensure!(
            batch == 1,
            "generate without cache currently supports batch size 1 (got {batch})"
        );
        if options.max_new_tokens == 0 {
            return self.empty_generation();
        }
        ensure!(
            options.position_ids.is_none(),
            "generate without cache requires position_ids to be computed internally"
        );

        let token_rows = input_ids
            .to_dtype(DType::I64)?
            .to_vec2::<i64>()
            .context("failed to extract input_ids for no-cache generation")?;
        let mut tokens = token_rows
            .into_iter()
            .next()
            .context("input_ids must have batch dimension 1")?;
        ensure!(
            tokens.len() == seq_len,
            "token vector length {} does not match seq_len {}",
            tokens.len(),
            seq_len
        );

        let mut attention_vec = if let Some(mask) = options.attention_mask {
            let rows = mask
                .to_dtype(DType::I64)?
                .to_vec2::<i64>()
                .context("failed to materialize attention mask for no-cache generation")?;
            let row = rows
                .into_iter()
                .next()
                .context("attention mask must have batch dimension 1")?;
            ensure!(
                row.len() == tokens.len(),
                "attention mask length {} does not match token count {}",
                row.len(),
                tokens.len()
            );
            Some(row)
        } else {
            None
        };

        let mut image_mask_vec = if let Some(mask) = options.images_seq_mask {
            let rows = mask
                .to_dtype(DType::U8)?
                .to_vec2::<u8>()
                .context("failed to materialize image mask for no-cache generation")?;
            let row = rows
                .into_iter()
                .next()
                .context("images_seq_mask must have batch dimension 1")?;
            ensure!(
                row.len() == tokens.len(),
                "images_seq_mask length {} does not match token count {}",
                row.len(),
                tokens.len()
            );
            Some(row)
        } else {
            None
        };

        let mut _owned_embeddings: Option<Vec<Tensor>> = None;
        let image_embeddings_slice: Option<&[Tensor]> =
            if let Some(slice) = options.image_embeddings {
                Some(slice)
            } else if let Some(inputs) = options.image_inputs {
                let computed = self.compute_image_embeddings(inputs)?;
                _owned_embeddings = Some(computed);
                _owned_embeddings.as_ref().map(|v| v.as_slice())
            } else {
                None
            };
        let forward_image_inputs = if image_embeddings_slice.is_some() {
            None
        } else {
            options.image_inputs
        };

        let to_tensor_i64 = |data: &[i64], device: &Device| -> Result<Tensor> {
            Ok(Tensor::from_vec(data.to_vec(), (1, data.len()), device)?.to_dtype(DType::I64)?)
        };
        let to_tensor_u8 = |data: &[u8], device: &Device| -> Result<Tensor> {
            Ok(Tensor::from_vec(data.to_vec(), (1, data.len()), device)?.to_dtype(DType::U8)?)
        };

        let mut attention_tensor = match &attention_vec {
            Some(vec) => Some(to_tensor_i64(vec, self.device())?),
            None => None,
        };
        let mut image_mask_tensor = match &image_mask_vec {
            Some(vec) => Some(to_tensor_u8(vec, self.device())?),
            None => None,
        };

        let input_tensor =
            to_tensor_i64(&tokens, self.device()).context("failed to build prefill tokens")?;
        let prefill = self.forward(
            Some(&input_tensor),
            None,
            attention_tensor.as_ref(),
            None,
            image_mask_tensor.as_ref(),
            forward_image_inputs,
            image_embeddings_slice,
            None,
            false,
        )?;
        let logits = prefill
            .logits
            .get(0)
            .context("prefill logits missing batch dimension")?
            .get(tokens.len() - 1)
            .context("prefill logits missing final timestep")?;
        let mut current = self.select_token_id(&logits)?;
        if let Some(eos) = options.eos_token_id {
            if current == eos {
                return self.empty_generation();
            }
        }

        let progress_callback = options.progress_callback;
        let mut generated = Vec::with_capacity(options.max_new_tokens);
        for step in 0..options.max_new_tokens {
            generated.push(current);
            if let Some(cb) = progress_callback {
                cb(generated.len(), &generated);
            }
            if step + 1 == options.max_new_tokens {
                break;
            }

            tokens.push(current);
            if let Some(mask) = image_mask_vec.as_mut() {
                mask.push(0);
            }
            if let Some(mask) = attention_vec.as_mut() {
                mask.push(1);
            }

            attention_tensor = match &attention_vec {
                Some(vec) => Some(to_tensor_i64(vec, self.device())?),
                None => None,
            };
            image_mask_tensor = match &image_mask_vec {
                Some(vec) => Some(to_tensor_u8(vec, self.device())?),
                None => None,
            };
            let input_tensor =
                to_tensor_i64(&tokens, self.device()).context("failed to build decode tokens")?;
            let forward = self.forward(
                Some(&input_tensor),
                None,
                attention_tensor.as_ref(),
                None,
                image_mask_tensor.as_ref(),
                forward_image_inputs,
                image_embeddings_slice,
                None,
                false,
            )?;
            let seq_pos = tokens.len() - 1;
            let next_logits = forward
                .logits
                .get(0)
                .context("decode logits missing batch dimension")?
                .get(seq_pos)
                .context("decode logits missing timestep")?;
            current = self.select_token_id(&next_logits)?;
            if let Some(eos) = options.eos_token_id {
                if current == eos {
                    break;
                }
            }
        }

        let len = generated.len();
        Ok(Tensor::from_vec(generated, (1, len), self.device())?.to_dtype(DType::I64)?)
    }

    fn empty_generation(&self) -> Result<Tensor> {
        Ok(Tensor::from_vec(Vec::<i64>::new(), (1, 0), self.device())?.to_dtype(DType::I64)?)
    }

    fn select_token_id(&self, logits: &Tensor) -> Result<i64> {
        let idx = logits.argmax(D::Minus1)?;
        let idx = if idx.dtype() == DType::I64 {
            idx
        } else {
            idx.to_dtype(DType::I64)?
        };
        idx.to_scalar::<i64>()
            .context("failed to convert argmax index to scalar")
    }
}

fn round_ties_to_even(value: f64) -> f64 {
    let rounded = value.round();
    if (value - rounded).abs() != 0.5 {
        return rounded;
    }
    let truncated = value.trunc();
    if truncated as i64 % 2 == 0 {
        truncated
    } else {
        truncated + value.signum()
    }
}

pub fn build_global_view(image: &DynamicImage, base_size: u32) -> DynamicImage {
    let mean = (0.5 * 255.0) as u8;
    let mut canvas = RgbImage::from_pixel(base_size, base_size, Rgb([mean, mean, mean]));
    let (orig_w, orig_h) = image.dimensions();
    if orig_w == 0 || orig_h == 0 {
        return DynamicImage::ImageRgb8(canvas);
    }
    let scale = (base_size as f64 / orig_w as f64).min(base_size as f64 / orig_h as f64);
    let new_w = round_ties_to_even(orig_w as f64 * scale)
        .max(1.0)
        .min(base_size as f64) as u32;
    let new_h = round_ties_to_even(orig_h as f64 * scale)
        .max(1.0)
        .min(base_size as f64) as u32;

    let rgb_image = image.to_rgb8();
    let resized = resize_bicubic(&rgb_image, new_w, new_h);

    let x_off = round_ties_to_even((base_size as f64 - new_w as f64) * 0.5) as i64;
    let y_off = round_ties_to_even((base_size as f64 - new_h as f64) * 0.5) as i64;
    imageops::replace(&mut canvas, &resized, x_off, y_off);
    DynamicImage::ImageRgb8(canvas)
}

pub fn image_to_tensor(image: &DynamicImage, device: &Device, dtype: DType) -> Result<Tensor> {
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let value = rgb.get_pixel(x, y)[c as usize] as f32 / 255.0;
                let normalized = (value - 0.5) / 0.5;
                data.push(normalized);
            }
        }
    }
    let tensor = Tensor::from_vec(data, (3, height as usize, width as usize), device)?;
    if tensor.dtype() == dtype {
        Ok(tensor)
    } else {
        Ok(tensor.to_dtype(dtype)?)
    }
}
