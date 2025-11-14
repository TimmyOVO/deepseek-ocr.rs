use anyhow::{Context, Result, ensure};
use candle_core::{DType, Module, Tensor, shape::D};
use candle_nn::{
    Conv2dConfig, LayerNorm, VarBuilder, conv2d_no_bias, layer_norm,
    ops::{sigmoid, softmax},
};

use crate::config::{DeepseekOcrConfig, VisionBackboneConfig};
use crate::transformer::weights::{LinearWeights, qualified_name};

/// Hyper-parameters describing the CLIP-L vision transformer used by DeepSeek-OCR.
#[derive(Debug, Clone)]
pub struct ClipVisionParams {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub ffn_hidden_size: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub seq_length: usize,
    pub layernorm_epsilon: f64,
}

impl ClipVisionParams {
    pub fn from_config(cfg: &DeepseekOcrConfig) -> Result<Self> {
        let clip_cfg = cfg
            .vision_config
            .as_ref()
            .and_then(|vision| vision.width.get("clip-l-14-224"))
            .cloned()
            .context("clip-l-14-224 vision backbone missing from config")?;
        Self::from_backbone(&clip_cfg)
    }

    fn from_backbone(cfg: &VisionBackboneConfig) -> Result<Self> {
        let hidden_size = cfg.width.context("clip hidden size missing")?;
        let num_heads = cfg.heads.context("clip num_heads missing")?;
        let num_layers = cfg.layers.context("clip num_layers missing")?;
        let patch_size = cfg.patch_size.context("clip patch_size missing")?;
        let image_size = cfg.image_size.context("clip image_size missing")?;
        let seq_length = (image_size / patch_size) * (image_size / patch_size);
        Ok(Self {
            hidden_size,
            num_layers,
            num_heads,
            ffn_hidden_size: hidden_size * 4,
            image_size,
            patch_size,
            seq_length,
            layernorm_epsilon: 1e-5,
        })
    }
}

/// CLIP-L transformer stack used to fuse SAM patch embeddings with learned positional context.
pub struct ClipVisionModel {
    params: ClipVisionParams,
    embeddings: ClipEmbeddings,
    transformer: ClipTransformer,
    pre_layernorm: LayerNorm,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone)]
pub struct ClipDebugTrace {
    pub embeddings: Tensor,
    pub pre_layernorm: Tensor,
    pub layer_outputs: Vec<Tensor>,
    pub output: Tensor,
}

impl ClipVisionModel {
    pub fn load(cfg: &DeepseekOcrConfig, vb: &VarBuilder) -> Result<Self> {
        let params = ClipVisionParams::from_config(cfg)?;
        let embeddings = ClipEmbeddings::load(&params, vb.pp("embeddings"))?;
        let transformer = ClipTransformer::load(&params, &vb.pp("transformer"))?;
        let pre_layernorm = layer_norm(
            params.hidden_size,
            params.layernorm_epsilon,
            vb.pp("pre_layrnorm"),
        )?;
        Ok(Self {
            params,
            embeddings,
            transformer,
            pre_layernorm,
        })
    }

    pub fn params(&self) -> &ClipVisionParams {
        &self.params
    }

    /// Forward pass through embeddings + transformer stack.
    ///
    /// * `pixel_values` – images with shape `[batch, 3, H, W]`. When `patch_embeds` is supplied,
    ///   it should contain precomputed patch projections with shape `[batch, hidden, grid, grid]`.
    pub fn forward(&self, pixel_values: &Tensor, patch_embeds: Option<&Tensor>) -> Result<Tensor> {
        let embeds = self.embeddings.forward(pixel_values, patch_embeds)?;
        let hidden = self.pre_layernorm.forward(&embeds)?;
        self.transformer.forward(&hidden)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn forward_with_trace(
        &self,
        pixel_values: &Tensor,
        patch_embeds: Option<&Tensor>,
    ) -> Result<ClipDebugTrace> {
        let embeddings = self.embeddings.forward(pixel_values, patch_embeds)?;
        let pre_layernorm = self.pre_layernorm.forward(&embeddings)?;
        let (output, layers) = self.transformer.forward_with_trace(&pre_layernorm)?;
        Ok(ClipDebugTrace {
            embeddings,
            pre_layernorm,
            layer_outputs: layers,
            output,
        })
    }
}

struct ClipEmbeddings {
    class_embedding: Tensor,
    patch_embedding: Option<candle_nn::Conv2d>,
    position_embedding: Tensor,
    seq_length: usize,
}

impl ClipEmbeddings {
    fn load(params: &ClipVisionParams, vb: VarBuilder) -> Result<Self> {
        let class_embedding = vb
            .get(params.hidden_size, "class_embedding")
            .context("missing clip class_embedding")?;
        let patch_embedding = if vb.contains_tensor("patch_embedding.weight") {
            let mut config = Conv2dConfig::default();
            config.stride = params.patch_size;
            config.padding = 0;
            let conv = conv2d_no_bias(
                3,
                params.hidden_size,
                params.patch_size,
                config,
                vb.pp("patch_embedding"),
            )?;
            Some(conv)
        } else {
            None
        };
        let position_embedding = vb
            .get(
                (params.seq_length + 1, params.hidden_size),
                "position_embedding.weight",
            )
            .context("missing clip position_embedding")?;
        Ok(Self {
            class_embedding,
            patch_embedding,
            position_embedding,
            seq_length: params.seq_length,
        })
    }

    fn forward(&self, pixel_values: &Tensor, patch_embeds: Option<&Tensor>) -> Result<Tensor> {
        let (batch, channels, height, width) = pixel_values
            .shape()
            .dims4()
            .context("clip pixel values must have shape [batch, 3, H, W]")?;
        ensure!(channels == 3, "clip expects 3-channel inputs");
        let patch = match patch_embeds {
            Some(patch) => patch.clone(),
            None => {
                let conv = self
                    .patch_embedding
                    .as_ref()
                    .context("patch_embeds missing and patch_embedding weights unavailable")?;
                conv.forward(pixel_values)?
            }
        };

        let (patch_batch, embed_dim, grid_h, grid_w) = patch
            .shape()
            .dims4()
            .context("clip patch embeddings must be 4D")?;
        ensure!(patch_batch == batch, "patch batch mismatch");
        ensure!(
            grid_h == grid_w,
            "clip patch grid must be square (got {}x{})",
            grid_h,
            grid_w
        );
        let num_patches = grid_h * grid_w;

        let patches = patch
            .reshape((batch, embed_dim, num_patches))?
            .transpose(1, 2)?
            .contiguous()?;
        let class_embedding = self
            .class_embedding
            .reshape((1, 1, embed_dim))?
            .expand((batch, 1, embed_dim))?;
        let tokens = Tensor::cat(&[class_embedding, patches], 1)?;

        let base_pos = self
            .position_embedding
            .reshape((1, self.seq_length + 1, embed_dim))?;
        let pos = adapt_position_embedding(&base_pos, num_patches + 1)?;
        let tokens = tokens
            .add(&pos.expand((batch, num_patches + 1, embed_dim))?)?
            .contiguous()?;

        ensure!(
            height % grid_h == 0 && width % grid_w == 0,
            "clip image dims {}x{} must align with patch grid {}",
            height,
            width,
            grid_h
        );
        Ok(tokens)
    }
}

struct ClipTransformer {
    layers: Vec<ClipBlock>,
}

impl ClipTransformer {
    fn load(params: &ClipVisionParams, vb: &VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(params.num_layers);
        let layers_vb = vb.pp("layers");
        for idx in 0..params.num_layers {
            layers.push(ClipBlock::load(params, &layers_vb.pp(&idx.to_string()))?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden = hidden_states.clone();
        for block in &self.layers {
            hidden = block.forward(&hidden)?;
        }
        Ok(hidden)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn forward_with_trace(&self, hidden_states: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let mut hidden = hidden_states.clone();
        let mut traces = Vec::with_capacity(self.layers.len());
        for block in &self.layers {
            hidden = block.forward(&hidden)?;
            traces.push(hidden.clone());
        }
        Ok((hidden, traces))
    }
}

struct ClipBlock {
    ln1: LayerNorm,
    attn: ClipAttention,
    ln2: LayerNorm,
    mlp: ClipMlp,
}

impl ClipBlock {
    fn load(params: &ClipVisionParams, vb: &VarBuilder) -> Result<Self> {
        let ln1 = layer_norm(
            params.hidden_size,
            params.layernorm_epsilon,
            vb.pp("layer_norm1"),
        )?;
        let attn = ClipAttention::load(params, &vb.pp("self_attn"))?;
        let ln2 = layer_norm(
            params.hidden_size,
            params.layernorm_epsilon,
            vb.pp("layer_norm2"),
        )?;
        let mlp = ClipMlp::load(params, &vb.pp("mlp"))?;
        Ok(Self {
            ln1,
            attn,
            ln2,
            mlp,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let normed = self.ln1.forward(hidden)?;
        let attn = self.attn.forward(&normed)?;
        let hidden = hidden.add(&attn)?;
        let normed = self.ln2.forward(&hidden)?;
        let mlp = self.mlp.forward(&normed)?;
        Ok(hidden.add(&mlp)?)
    }
}

struct ClipAttention {
    num_heads: usize,
    head_dim: usize,
    qkv_proj: LinearWeights,
    out_proj: LinearWeights,
}

impl ClipAttention {
    fn load(params: &ClipVisionParams, vb: &VarBuilder) -> Result<Self> {
        let num_heads = params.num_heads;
        let head_dim = params.hidden_size / num_heads;
        ensure!(
            head_dim * num_heads == params.hidden_size,
            "hidden size {} not divisible by heads {}",
            params.hidden_size,
            num_heads
        );
        let qkv_proj = load_linear(
            &vb.pp("qkv_proj"),
            params.hidden_size * 3,
            params.hidden_size,
            true,
        )?;
        let out_proj = load_linear(
            &vb.pp("out_proj"),
            params.hidden_size,
            params.hidden_size,
            true,
        )?;
        Ok(Self {
            num_heads,
            head_dim,
            qkv_proj,
            out_proj,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let qkv = apply_linear(hidden, &self.qkv_proj)?;
        let (batch, seq_len, dim) = qkv.shape().dims3()?;
        let expected = self.num_heads * self.head_dim;
        ensure!(
            dim == expected * 3,
            "clip qkv projection produced dim {}, expected {}",
            dim,
            expected * 3
        );
        let q = qkv
            .narrow(D::Minus1, 0, expected)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let k = qkv
            .narrow(D::Minus1, expected, expected)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = qkv
            .narrow(D::Minus1, expected * 2, expected)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt() as f32;
        let attn = scaled_dot_product_attention(&q, &k, &v, scale)?;
        let attn =
            attn.transpose(1, 2)?
                .reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        apply_linear(&attn, &self.out_proj)
    }
}

struct ClipMlp {
    fc1: LinearWeights,
    fc2: LinearWeights,
}

impl ClipMlp {
    fn load(params: &ClipVisionParams, vb: &VarBuilder) -> Result<Self> {
        let fc1 = load_linear(
            &vb.pp("fc1"),
            params.ffn_hidden_size,
            params.hidden_size,
            true,
        )?;
        let fc2 = load_linear(
            &vb.pp("fc2"),
            params.hidden_size,
            params.ffn_hidden_size,
            true,
        )?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let x = apply_linear(hidden, &self.fc1)?;
        let x = quick_gelu(&x)?;
        apply_linear(&x, &self.fc2)
    }
}

fn quick_gelu(x: &Tensor) -> Result<Tensor> {
    let inner = x.affine(1.702, 0.0)?;
    Ok(sigmoid(&inner)?.mul(x)?)
}

fn apply_linear(input: &Tensor, weights: &LinearWeights) -> Result<Tensor> {
    let dims = input.shape().dims();
    ensure!(dims.len() >= 2, "linear expects rank >= 2");
    let last_dim = *dims.last().expect("at least one dim");
    let (out_dim, in_dim) = (weights.out_dim, weights.in_dim);
    ensure!(
        in_dim == last_dim,
        "linear weight expects input dim {}, got {}",
        in_dim,
        last_dim
    );
    let leading = dims[..dims.len() - 1].iter().product::<usize>();
    let input2d = input.reshape((leading, in_dim))?;
    let weight = weights
        .weight
        .as_ref()
        .context("vision linear weights require float tensors")?;
    let mut proj = input2d.matmul(&weight.transpose(0, 1)?)?;
    if let Some(bias) = &weights.bias {
        proj = proj.broadcast_add(&bias.reshape((1, out_dim))?)?;
    }
    proj.reshape(
        dims[..dims.len() - 1]
            .iter()
            .copied()
            .chain(std::iter::once(out_dim))
            .collect::<Vec<_>>(),
    )
    .context("reshape linear output")
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale as f64, 0.0)?;
    let attn = softmax(&scores, D::Minus1)?;
    Ok(attn.matmul(v)?)
}

fn load_linear(
    vb: &VarBuilder,
    out_dim: usize,
    in_dim: usize,
    bias: bool,
) -> Result<LinearWeights> {
    let label = qualified_name(vb, "weight");
    let weight = vb
        .get((out_dim, in_dim), "weight")
        .with_context(|| "missing linear weight")?
        .contiguous()?;
    let bias = if bias && vb.contains_tensor("bias") {
        Some(
            vb.get(out_dim, "bias")
                .with_context(|| "missing linear bias")?
                .contiguous()?,
        )
    } else {
        None
    };
    Ok(LinearWeights {
        weight: Some(weight),
        bias,
        qmatmul: None,
        out_dim,
        in_dim,
        label,
    })
}

fn adapt_position_embedding(table: &Tensor, target_tokens: usize) -> Result<Tensor> {
    let (batch, src_tokens, hidden) = table
        .shape()
        .dims3()
        .context("clip positional table must be 3D")?;
    ensure!(batch == 1, "clip positional table expects batch size 1");
    if src_tokens == target_tokens {
        return Ok(table.clone());
    }
    ensure!(
        src_tokens > 1 && target_tokens > 1,
        "clip positional table requires patch tokens"
    );
    let src_patches = src_tokens - 1;
    let tgt_patches = target_tokens - 1;
    let src_size =
        integer_sqrt(src_patches).context("clip positional table src tokens not square")?;
    let tgt_size =
        integer_sqrt(tgt_patches).context("clip positional table tgt tokens not square")?;

    let cls_token = table.narrow(D::Minus2, 0, 1)?.contiguous()?;
    let patch_tokens = table
        .narrow(D::Minus2, 1, src_patches)?
        .contiguous()
        .context("clip positional patch slice not contiguous")?;
    let patch_grid = patch_tokens
        .reshape((src_size, src_size, hidden))?
        .permute((2, 0, 1))?
        .unsqueeze(0)?
        .contiguous()
        .context("clip positional grid reshape not contiguous")?;
    let float_grid = if patch_grid.dtype() == DType::F32 {
        patch_grid.clone()
    } else {
        patch_grid.to_dtype(DType::F32)?
    };
    let resized = crate::vision::sam::bicubic_resize_antialiased(&float_grid, tgt_size, tgt_size)?;
    let resized = if resized.dtype() == patch_grid.dtype() {
        resized
    } else {
        resized.to_dtype(patch_grid.dtype())?
    };
    let resized_tokens = resized
        .squeeze(0)?
        .permute((1, 2, 0))?
        .reshape((tgt_patches, hidden))?
        .contiguous()
        .context("clip positional resized tokens not contiguous")?;
    let cls_token = if cls_token.dtype() == resized_tokens.dtype() {
        cls_token
    } else {
        cls_token.to_dtype(resized_tokens.dtype())?
    };
    let cls_token = cls_token.reshape((1, hidden))?;
    let combined = Tensor::cat(&[cls_token, resized_tokens], 0)?;
    combined
        .reshape((1, target_tokens, hidden))
        .context("clip positional combined reshape failed")
}

#[doc(hidden)]
pub fn adapt_position_embedding_for_tests(table: &Tensor, target_tokens: usize) -> Result<Tensor> {
    adapt_position_embedding(table, target_tokens)
}

fn integer_sqrt(value: usize) -> Option<usize> {
    let root = (value as f64).sqrt().round() as usize;
    if root * root == value {
        Some(root)
    } else if (root + 1) * (root + 1) == value {
        Some(root + 1)
    } else if root > 0 && (root - 1) * (root - 1) == value {
        Some(root - 1)
    } else {
        None
    }
}
