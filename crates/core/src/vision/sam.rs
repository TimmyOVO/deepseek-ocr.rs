use crate::config::{DeepseekOcrConfig, VisionBackboneConfig, VisionConfig};
use anyhow::{Context, Result, anyhow, bail, ensure};
use candle_core::{DType, Device, Module, Tensor, shape::D};
use candle_nn::{
    Conv2d, Conv2dConfig, LayerNorm, VarBuilder, conv2d, conv2d_no_bias, layer_norm, ops::softmax,
};
use std::path::Path;

/// Parameter bundle for the SAM ViT backbone.
#[derive(Debug, Clone)]
pub struct SamBackboneParams {
    pub image_size: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub window_size: usize,
    pub neck_channels: usize,
    pub out_channels: Vec<usize>,
    pub global_attn_indexes: Vec<usize>,
    pub mlp_ratio: f64,
    pub use_rel_pos: bool,
    pub use_abs_pos: bool,
    pub qkv_bias: bool,
    pub norm_eps: f64,
}

impl SamBackboneParams {
    /// Build parameters from the Hugging Face config (vision_config.width.sam_vit_b).
    pub fn from_config(cfg: &DeepseekOcrConfig) -> Result<Self> {
        let base = cfg
            .vision_config
            .as_ref()
            .and_then(|vision| vision.width.get("sam_vit_b"))
            .cloned()
            .ok_or_else(|| anyhow!("sam_vit_b vision backbone missing from config"))?;
        Self::from_backbone_cfg(&base, cfg.vision_config.as_ref())
    }

    fn from_backbone_cfg(
        backbone: &VisionBackboneConfig,
        vision: Option<&VisionConfig>,
    ) -> Result<Self> {
        let image_size = vision
            .and_then(|v| v.image_size)
            .or(backbone.image_size)
            .unwrap_or(1024);
        let patch_size = backbone.patch_size.unwrap_or(16);
        let embed_dim = backbone.width.unwrap_or(768);
        let depth = backbone.layers.unwrap_or(12);
        let num_heads = backbone.heads.unwrap_or(12);
        let window_size = 14;
        let neck_channels = 256;
        let out_channels = backbone
            .downsample_channels
            .clone()
            .unwrap_or_else(|| vec![512, 1024]);
        let global_attn_indexes = backbone
            .global_attn_indexes
            .clone()
            .unwrap_or_else(|| vec![2, 5, 8, 11]);

        Ok(Self {
            image_size,
            patch_size,
            embed_dim,
            depth,
            num_heads,
            window_size,
            neck_channels,
            out_channels,
            global_attn_indexes,
            mlp_ratio: 4.0,
            use_rel_pos: true,
            use_abs_pos: true,
            qkv_bias: true,
            norm_eps: 1e-6,
        })
    }
}

/// Placeholder for the SAM ViT backbone implementation.
///
/// This currently holds configuration metadata only; the forward pass will be
/// implemented when Tensor kernels are wired up with Candle.
pub struct SamBackbone {
    pub params: SamBackboneParams,
    patch_embed: PatchEmbed,
    pos_embed: Option<Tensor>,
    blocks: Vec<SamBlock>,
    neck: SamNeck,
    downsample: SamDownsample,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone)]
pub struct SamDebugTrace {
    pub patch_embed: Tensor,
    pub pos_added: Option<Tensor>,
    pub block_outputs: Vec<Tensor>,
    pub neck_conv1: Tensor,
    pub neck_norm1: Tensor,
    pub neck_conv2: Tensor,
    pub neck_norm2: Tensor,
    pub net2: Tensor,
    pub net3: Tensor,
}

impl SamBackbone {
    pub fn new(cfg: &DeepseekOcrConfig, vb: &VarBuilder) -> Result<Self> {
        let params = SamBackboneParams::from_config(cfg)?;
        let patch_embed =
            PatchEmbed::new(3, params.embed_dim, params.patch_size, vb.pp("patch_embed"))
                .map_err(|err| anyhow!(err))?;
        let tokens_per_side = params.image_size / params.patch_size;
        let pos_embed = if vb.contains_tensor("pos_embed") {
            Some(
                vb.get(
                    (1, tokens_per_side, tokens_per_side, params.embed_dim),
                    "pos_embed",
                )?
                .contiguous()
                .context("pos_embed must be contiguous")?,
            )
        } else {
            None
        };

        let blocks_vb = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(params.depth);
        for idx in 0..params.depth {
            blocks.push(
                SamBlock::load(idx, &params, &blocks_vb.pp(idx.to_string()))
                    .context(format!("failed to load SAM block {idx}"))?,
            );
        }

        let neck = SamNeck::new(params.embed_dim, params.neck_channels, vb.pp("neck"))
            .map_err(|err| anyhow!(err))?;
        let downsample = SamDownsample::new(params.neck_channels, &params.out_channels, vb.clone())
            .map_err(|err| anyhow!(err))?;

        Ok(Self {
            params,
            patch_embed,
            pos_embed,
            blocks,
            neck,
            downsample,
        })
    }

    pub fn with_dummy_weights(cfg: &DeepseekOcrConfig) -> Result<Self> {
        let vb = VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu);
        Self::new(cfg, &vb)
    }

    /// Construct the SAM backbone by loading real weights from safetensors.
    ///
    /// # Safety
    ///
    /// Relies on memory-mapping the provided safetensor shard(s).
    pub unsafe fn from_safetensors<P: AsRef<Path>>(
        cfg: &DeepseekOcrConfig,
        paths: &[P],
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(paths, dtype, device)
                .context("failed to mmap safetensors for SAM backbone")?
        };
        let sam_vb = vb.pp("model").pp("sam_model");
        Self::new(cfg, &sam_vb)
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = input
            .shape()
            .dims4()
            .map_err(|_| anyhow!("expected input shape [batch, channels, height, width]"))?;

        if channels != 3 {
            bail!(
                "sam backbone expects 3-channel input, received {} channels",
                channels
            );
        }

        let _patch_shape = patch_embed_shape(
            batch,
            channels,
            height,
            width,
            self.params.patch_size,
            self.params.embed_dim,
        )?;

        let patch = self
            .patch_embed
            .forward(input)
            .map_err(|err| anyhow!("patch embedding failed: {err}"))?;

        let mut x = patch
            .permute((0, 2, 3, 1))?
            .contiguous()
            .context("sam patch permute not contiguous")?;

        let (_, tokens_h, tokens_w, channels) = x.shape().dims4()?;
        if let Some(pos_embed) = &self.pos_embed {
            let (_, _src_h, _src_w, _) = pos_embed.shape().dims4()?;
            // anyhow::ensure!(
            //     tokens_h == src_h && tokens_w == src_w,
            //     "SAM local grid {}x{} != pos_embed {}x{}; this triggers interpolation and large diffs",
            //     tokens_h,
            //     tokens_w,
            //     src_h,
            //     src_w
            // );
            let mut pos = adapt_position_embedding(pos_embed, tokens_h, tokens_w)?;
            if pos.shape().dims4()?.0 == 1 && batch > 1 {
                pos = pos.expand((batch, tokens_h, tokens_w, channels))?;
            }
            if pos.dtype() != x.dtype() {
                pos = pos.to_dtype(x.dtype())?;
            }
            x = x.add(&pos)?;
        }

        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        let x = x
            .permute((0, 3, 1, 2))?
            .contiguous()
            .context("sam transformer output permute not contiguous")?;

        let x = self
            .neck
            .forward(&x)
            .map_err(|err| anyhow!("neck forward failed: {err}"))?;

        let x = self
            .downsample
            .forward(x)
            .map_err(|err| anyhow!("downsample forward failed: {err}"))?;

        Ok(x)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn forward_with_trace(&self, input: &Tensor) -> Result<(Tensor, SamDebugTrace)> {
        let (batch, channels, height, width) = input
            .shape()
            .dims4()
            .map_err(|_| anyhow!("expected input shape [batch, channels, height, width]"))?;
        if channels != 3 {
            bail!(
                "sam backbone expects 3-channel input, received {} channels",
                channels
            );
        }

        let _patch_shape = patch_embed_shape(
            batch,
            channels,
            height,
            width,
            self.params.patch_size,
            self.params.embed_dim,
        )?;

        let patch = self
            .patch_embed
            .forward(input)
            .map_err(|err| anyhow!("patch embedding failed: {err}"))?;

        let mut x = patch
            .permute((0, 2, 3, 1))?
            .contiguous()
            .context("sam patch permute not contiguous")?;
        let patch_trace = x.clone();

        let mut pos_trace = None;
        {
            let (_, tokens_h, tokens_w, channels) = x.shape().dims4()?;
            if let Some(pos_embed) = &self.pos_embed {
                let mut pos = adapt_position_embedding(pos_embed, tokens_h, tokens_w)?;
                if pos.shape().dims4()?.0 == 1 && batch > 1 {
                    pos = pos.expand((batch, tokens_h, tokens_w, channels))?;
                }
                if pos.dtype() != x.dtype() {
                    pos = pos.to_dtype(x.dtype())?;
                }
                x = x.add(&pos)?;
                pos_trace = Some(x.clone());
            }
        }

        let mut block_traces = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            x = block.forward(&x)?;
            block_traces.push(x.clone());
        }

        let x = x
            .permute((0, 3, 1, 2))?
            .contiguous()
            .context("sam transformer output permute not contiguous")?;

        let conv1 = self
            .neck
            .conv1
            .forward(&x)
            .map_err(|err| anyhow!("neck conv1 forward failed: {err}"))?;
        let conv1_trace = conv1.clone();
        let norm1 = self
            .neck
            .norm1
            .forward(&conv1)
            .map_err(|err| anyhow!("neck norm1 forward failed: {err}"))?;
        let norm1_trace = norm1.clone();
        let conv2 = self
            .neck
            .conv2
            .forward(&norm1)
            .map_err(|err| anyhow!("neck conv2 forward failed: {err}"))?;
        let conv2_trace = conv2.clone();
        let norm2 = self
            .neck
            .norm2
            .forward(&conv2)
            .map_err(|err| anyhow!("neck norm2 forward failed: {err}"))?;
        let norm2_trace = norm2.clone();

        let (_, _, h_after, w_after) = norm2.shape().dims4()?;
        if h_after % 2 != 0 || w_after % 2 != 0 {
            bail!(
                "spatial dims {}x{} cannot be evenly downsampled by stride 2",
                h_after,
                w_after
            );
        }
        let net2 = self
            .downsample
            .net2
            .forward(&norm2)
            .map_err(|err| anyhow!("downsample net_2 forward failed: {err}"))?;
        let net2_trace = net2.clone();
        let (_, _, h_down, w_down) = net2.shape().dims4()?;
        if h_down % 2 != 0 || w_down % 2 != 0 {
            bail!(
                "spatial dims {}x{} cannot be evenly downsampled by stride 2",
                h_down,
                w_down
            );
        }
        let net3 = self
            .downsample
            .net3
            .forward(&net2)
            .map_err(|err| anyhow!("downsample net_3 forward failed: {err}"))?;
        let net3_trace = net3.clone();

        let trace = SamDebugTrace {
            patch_embed: patch_trace,
            pos_added: pos_trace,
            block_outputs: block_traces,
            neck_conv1: conv1_trace,
            neck_norm1: norm1_trace,
            neck_conv2: conv2_trace,
            neck_norm2: norm2_trace,
            net2: net2_trace,
            net3: net3_trace.clone(),
        };

        Ok((net3, trace))
    }
}

struct PatchEmbed {
    conv: Conv2d,
}

impl PatchEmbed {
    fn new(
        in_channels: usize,
        out_channels: usize,
        patch_size: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let mut config = Conv2dConfig::default();
        config.stride = patch_size;
        config.padding = 0;
        let conv = conv2d(in_channels, out_channels, patch_size, config, vb.pp("proj"))?;
        Ok(Self { conv })
    }

    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        self.conv.forward(input)
    }
}

struct LayerNorm2d {
    norm: LayerNorm,
}

impl LayerNorm2d {
    fn new(channels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let norm = layer_norm(channels, 1e-6, vb)?;
        Ok(Self { norm })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let nhwc = x.permute((0, 2, 3, 1))?;
        let y = self.norm.forward(&nhwc)?;
        y.permute((0, 3, 1, 2))
    }
}

struct SamNeck {
    conv1: Conv2d,
    norm1: LayerNorm2d,
    conv2: Conv2d,
    norm2: LayerNorm2d,
}

impl SamNeck {
    fn new(embed_dim: usize, neck_channels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let conv1_cfg = Conv2dConfig::default();
        let conv1 = conv2d_no_bias(embed_dim, neck_channels, 1, conv1_cfg, vb.pp("0"))?;
        let norm1 = LayerNorm2d::new(neck_channels, vb.pp("1"))?;

        let mut conv2_cfg = Conv2dConfig::default();
        conv2_cfg.padding = 1;
        let conv2 = conv2d_no_bias(neck_channels, neck_channels, 3, conv2_cfg, vb.pp("2"))?;
        let norm2 = LayerNorm2d::new(neck_channels, vb.pp("3"))?;

        Ok(Self {
            conv1,
            norm1,
            conv2,
            norm2,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.conv1.forward(x)?;
        let x = self.norm1.forward(&x)?;
        let x = self.conv2.forward(&x)?;
        self.norm2.forward(&x)
    }
}

struct SamDownsample {
    net2: Conv2d,
    net3: Conv2d,
}

impl SamDownsample {
    fn new(
        in_channels: usize,
        out_channels: &[usize],
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        if out_channels.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "expected exactly two downsample stages (net_2/net_3), found {}",
                out_channels.len()
            )));
        }
        let mut cfg = Conv2dConfig::default();
        cfg.stride = 2;
        cfg.padding = 1;
        let net2 = conv2d_no_bias(in_channels, out_channels[0], 3, cfg, vb.pp("net_2"))?;
        let net3 = conv2d_no_bias(out_channels[0], out_channels[1], 3, cfg, vb.pp("net_3"))?;
        Ok(Self { net2, net3 })
    }

    fn forward(&self, mut x: Tensor) -> candle_core::Result<Tensor> {
        let (_, _, h, w) = x.shape().dims4()?;
        if h % 2 != 0 || w % 2 != 0 {
            return Err(candle_core::Error::Msg(format!(
                "spatial dims {}x{} cannot be evenly downsampled by stride 2",
                h, w
            )));
        }
        x = self.net2.forward(&x)?;
        let (_, _, h, w) = x.shape().dims4()?;
        if h % 2 != 0 || w % 2 != 0 {
            return Err(candle_core::Error::Msg(format!(
                "spatial dims {}x{} cannot be evenly downsampled by stride 2",
                h, w
            )));
        }
        self.net3.forward(&x)
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct PatchShape {
    tokens_h: usize,
    tokens_w: usize,
}

fn patch_embed_shape(
    _batch: usize,
    _channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    _embed_dim: usize,
) -> Result<PatchShape> {
    if height % patch_size != 0 || width % patch_size != 0 {
        bail!(
            "image dimensions {}x{} must be divisible by patch size {}",
            height,
            width,
            patch_size
        );
    }
    Ok(PatchShape {
        tokens_h: height / patch_size,
        tokens_w: width / patch_size,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct WindowPartitionShape {
    pub padded_height: usize,
    pub padded_width: usize,
    pub tiles_h: usize,
    pub tiles_w: usize,
}

pub fn window_partition_shape(h: usize, w: usize, window: usize) -> WindowPartitionShape {
    let pad_h = (window - h % window) % window;
    let pad_w = (window - w % window) % window;
    let padded_height = h + pad_h;
    let padded_width = w + pad_w;
    let tiles_h = padded_height / window;
    let tiles_w = padded_width / window;
    WindowPartitionShape {
        padded_height,
        padded_width,
        tiles_h,
        tiles_w,
    }
}

#[derive(Clone)]
struct LinearLayer {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl LinearLayer {
    fn load(vb: &VarBuilder, out_dim: usize, in_dim: usize, bias: bool) -> Result<Self> {
        let weight = vb
            .get((out_dim, in_dim), "weight")
            .with_context(|| "missing linear weight `weight`")?;
        let weight = weight
            .contiguous()
            .context("linear weight must be contiguous")?;
        let bias = if bias && vb.contains_tensor("bias") {
            Some(
                vb.get(out_dim, "bias")
                    .with_context(|| "missing linear bias `bias`")?,
            )
        } else {
            None
        };
        Ok(Self { weight, bias })
    }
}

fn linear_forward(layer: &LinearLayer, input: &Tensor) -> Result<Tensor> {
    let dims = input.shape().dims();
    anyhow::ensure!(dims.len() >= 2, "linear expects rank >= 2");
    let last_dim = *dims
        .last()
        .expect("linear expects rank >= 2 so last dim exists");
    let (out_dim, in_dim) = layer
        .weight
        .shape()
        .dims2()
        .context("linear weight must be 2D")?;
    anyhow::ensure!(
        in_dim == last_dim,
        "linear weight expects input dim {}, got {}",
        in_dim,
        last_dim
    );
    let leading = dims[..dims.len() - 1].iter().product::<usize>();
    let reshaped = input.reshape((leading, in_dim))?;
    let weight_t = if layer.weight.dtype() == reshaped.dtype() {
        layer.weight.transpose(0, 1)?
    } else {
        layer.weight.to_dtype(reshaped.dtype())?.transpose(0, 1)?
    };
    let mut output = reshaped.matmul(&weight_t)?;
    if let Some(bias) = &layer.bias {
        let bias = if bias.dtype() == output.dtype() {
            bias.reshape((1, out_dim))?
        } else {
            bias.to_dtype(output.dtype())?.reshape((1, out_dim))?
        };
        output = output.broadcast_add(&bias)?;
    }
    if output.dtype() != input.dtype() {
        output = output.to_dtype(input.dtype())?;
    }
    output
        .reshape(
            dims[..dims.len() - 1]
                .iter()
                .copied()
                .chain(std::iter::once(out_dim))
                .collect::<Vec<_>>(),
        )
        .context("linear output reshape failed")
}

struct SamBlock {
    norm1: LayerNorm,
    attn: SamAttention,
    norm2: LayerNorm,
    mlp: SamMlp,
    window_size: usize,
}

impl SamBlock {
    fn load(index: usize, params: &SamBackboneParams, vb: &VarBuilder) -> Result<Self> {
        let window_size = if params.global_attn_indexes.contains(&index) {
            0
        } else {
            params.window_size
        };
        let norm1 = layer_norm(params.embed_dim, params.norm_eps, vb.pp("norm1"))?;
        let attn = SamAttention::load(params, window_size, &vb.pp("attn"))?;
        let norm2 = layer_norm(params.embed_dim, params.norm_eps, vb.pp("norm2"))?;
        let mlp = SamMlp::load(params, &vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (_batch, height, width, _channels) = input.shape().dims4()?;
        let normed = self.norm1.forward(input)?;
        let attn_out = if self.window_size > 0 {
            let (windows, pad_hw) = window_partition(&normed, self.window_size)?;
            let attn_windows = self
                .attn
                .forward(&windows, (self.window_size, self.window_size))?;
            window_unpartition(&attn_windows, self.window_size, pad_hw, (height, width))?
        } else {
            self.attn.forward(&normed, (height, width))?
        };

        let residual = input.add(&attn_out)?;
        let normed2 = self.norm2.forward(&residual)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        Ok(residual.add(&mlp_out)?)
    }
}

struct SamAttention {
    num_heads: usize,
    head_dim: usize,
    qkv: LinearLayer,
    proj: LinearLayer,
    use_rel_pos: bool,
    rel_pos_h: Option<Tensor>,
    rel_pos_w: Option<Tensor>,
}

impl SamAttention {
    fn load(params: &SamBackboneParams, window_size: usize, vb: &VarBuilder) -> Result<Self> {
        let qkv_vb = vb.pp("qkv");
        let proj_vb = vb.pp("proj");
        let qkv = LinearLayer::load(&qkv_vb, params.embed_dim * 3, params.embed_dim, true)?;
        let proj = LinearLayer::load(&proj_vb, params.embed_dim, params.embed_dim, true)?;
        let input_tokens = if window_size > 0 {
            window_size
        } else {
            params.image_size / params.patch_size
        };
        let use_rel_pos = params.use_rel_pos && vb.contains_tensor("rel_pos_h");
        let rel_dim = 2 * input_tokens - 1;
        let head_dim = params.embed_dim / params.num_heads;
        let rel_pos_h = if use_rel_pos {
            Some(
                vb.get((rel_dim, head_dim), "rel_pos_h")?
                    .contiguous()
                    .context("rel_pos_h must be contiguous")?,
            )
        } else {
            None
        };
        let rel_pos_w = if use_rel_pos {
            Some(
                vb.get((rel_dim, head_dim), "rel_pos_w")?
                    .contiguous()
                    .context("rel_pos_w must be contiguous")?,
            )
        } else {
            None
        };
        Ok(Self {
            num_heads: params.num_heads,
            head_dim: params.embed_dim / params.num_heads,
            qkv,
            proj,
            use_rel_pos,
            rel_pos_h,
            rel_pos_w,
        })
    }

    fn forward(&self, input: &Tensor, spatial: (usize, usize)) -> Result<Tensor> {
        let (batch, height, width, _) = input.shape().dims4()?;
        let dtype = input.dtype();
        let seq_len = height * width;
        // qkv: (B, S, 3, H, Dh) —— 注意顺序：先 3，再 H
        let qkv = linear_forward(&self.qkv, input)?
            .reshape((batch, seq_len, 3, self.num_heads, self.head_dim))?
            .contiguous()
            .context("qkv reshape not contiguous")?;

        // 取 q/k/v: 先沿着 size=3 的维度切，再 permute 到 (B, H, S, Dh)
        // 维度记住：squeeze 后是 (B, S, H, Dh) → permute(0, 2, 1, 3)
        let q = qkv
            .narrow(D::Minus(3), 0, 1)?
            .squeeze(D::Minus(3))?
            .permute((0, 2, 1, 3))? // <-- 正确：0,2,1,3
            .contiguous()
            .context("q not contiguous")?;

        let k = qkv
            .narrow(D::Minus(3), 1, 1)?
            .squeeze(D::Minus(3))?
            .permute((0, 2, 1, 3))? // <-- 同上
            .contiguous()
            .context("k not contiguous")?;

        let v = qkv
            .narrow(D::Minus(3), 2, 1)?
            .squeeze(D::Minus(3))?
            .permute((0, 2, 1, 3))? // <-- 同上
            .contiguous()
            .context("v not contiguous")?;

        // matmul 前转成 f32，k^T 也要做 contiguous
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let (_b, h, _s, dh) = q.shape().dims4()?;
        ensure!(
            h == self.num_heads && dh == self.head_dim,
            "Q shape mismatch: got H={},Dh={}, expect H={},Dh={}",
            h,
            dh,
            self.num_heads,
            self.head_dim
        );
        let mut scores = q.matmul(
            &k.transpose(2, 3)?
                .contiguous()
                .context("k^T not contiguous")?,
        )?;
        scores = scores.affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;

        // 相对位置偏置
        if self.use_rel_pos {
            let mut attn_bias = compute_relative_bias(
                &q,
                spatial,
                self.rel_pos_h.as_ref(),
                self.rel_pos_w.as_ref(),
            )?;
            if attn_bias.dtype() != scores.dtype() {
                attn_bias = attn_bias.to_dtype(scores.dtype())?;
            }
            scores = scores.add(&attn_bias)?;
        }

        let attn = softmax(&scores, D::Minus1)?; // (B, H, S, S)
        let context = attn.matmul(&v)?; // (B, H, S, Dh)

        // 还原 → 线性映射
        let context = context
            .permute((0, 2, 1, 3))? // (B, S, H, Dh)
            .contiguous()
            .context("context not contiguous")?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?
            .contiguous()
            .context("context flat not contiguous")?;
        let context = context
            .reshape((batch, height, width, self.num_heads * self.head_dim))?
            .contiguous()
            .context("context hwc not contiguous")?;
        let out = linear_forward(&self.proj, &context)?;
        Ok(out.to_dtype(dtype)?)
    }
}

struct SamMlp {
    fc1: LinearLayer,
    fc2: LinearLayer,
}

impl SamMlp {
    fn load(params: &SamBackboneParams, vb: &VarBuilder) -> Result<Self> {
        let hidden = (params.embed_dim as f64 * params.mlp_ratio) as usize;
        let fc1_vb = if vb.contains_tensor("fc1.weight") {
            vb.pp("fc1")
        } else if vb.contains_tensor("lin1.weight") {
            vb.pp("lin1")
        } else {
            bail!("sam mlp missing fc1/lin1 weights")
        };
        let fc2_vb = if vb.contains_tensor("fc2.weight") {
            vb.pp("fc2")
        } else if vb.contains_tensor("lin2.weight") {
            vb.pp("lin2")
        } else {
            bail!("sam mlp missing fc2/lin2 weights")
        };
        let fc1 = LinearLayer::load(&fc1_vb, hidden, params.embed_dim, true)?;
        let fc2 = LinearLayer::load(&fc2_vb, params.embed_dim, hidden, true)?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = linear_forward(&self.fc1, input)?;
        let x = x.gelu()?;
        linear_forward(&self.fc2, &x)
    }
}

fn window_partition(x: &Tensor, window: usize) -> Result<(Tensor, (usize, usize))> {
    let (batch, height, width, channels) = x.shape().dims4()?;
    let pad_h = (window - height % window) % window;
    let pad_w = (window - width % window) % window;
    let dtype = x.dtype();
    let device = x.device();
    let padded = if pad_h == 0 && pad_w == 0 {
        x.clone()
    } else {
        let mut padded = Tensor::zeros(
            (batch, height + pad_h, width + pad_w, channels),
            dtype,
            device,
        )?;
        padded = padded.slice_assign(&[0..batch, 0..height, 0..width, 0..channels], x)?;
        padded
    };
    let hp = height + pad_h;
    let wp = width + pad_w;
    let windows = padded
        .reshape((batch, hp / window, window, wp / window, window, channels))?
        .permute((0, 1, 3, 2, 4, 5))?
        .reshape((
            batch * (hp / window) * (wp / window),
            window,
            window,
            channels,
        ))?;
    Ok((windows, (hp, wp)))
}

fn window_unpartition(
    windows: &Tensor,
    window: usize,
    pad_hw: (usize, usize),
    original_hw: (usize, usize),
) -> Result<Tensor> {
    let (hp, wp) = pad_hw;
    let (h, w) = original_hw;
    let (num_windows, _, _, channels) = windows.shape().dims4()?;
    let tiles = (hp / window) * (wp / window);
    let batch = num_windows / tiles;
    let restored = windows
        .reshape((batch, hp / window, wp / window, window, window, channels))?
        .permute((0, 1, 3, 2, 4, 5))?
        .reshape((batch, hp, wp, channels))?;
    if hp == h && wp == w {
        Ok(restored)
    } else {
        let trimmed = restored.narrow(1, 0, h)?.narrow(2, 0, w)?;
        Ok(trimmed
            .contiguous()
            .context("window unpartition slice not contiguous")?)
    }
}

fn adapt_position_embedding(
    pos_embed: &Tensor,
    target_h: usize,
    target_w: usize,
) -> Result<Tensor> {
    let pos = pos_embed
        .to_dtype(DType::F32)?
        .contiguous()
        .context("position embedding not contiguous")?;
    let (_, src_h, src_w, _hidden) = pos.shape().dims4()?;
    if src_h == target_h && src_w == target_w {
        return Ok(pos);
    }
    let permuted = pos.permute((0, 3, 1, 2))?;
    let resized = bicubic_resize_antialiased(&permuted, target_h, target_w)?;
    Ok(resized.permute((0, 2, 3, 1))?)
}

pub fn bicubic_resize_antialiased(input: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    #[inline(always)]
    fn cubic(a: f32, x: f32) -> f32 {
        let x = x.abs();
        if x <= 1.0 {
            (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
        } else if x < 2.0 {
            a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
        } else {
            0.0
        }
    }

    fn compute_axis_weights(
        in_len: usize,
        out_len: usize,
        scale: f32,
        support_scale: f32,
        a: f32,
    ) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
        // PyTorch 在下采样时会把 bicubic 核的支撑区放大到 2 * support_scale，
        // 相当于先低通滤波再重采样，避免 aliasing，尽量缩小和PyTorch实现的差别。
        let radius = 2.0 * support_scale;
        let mut weights = vec![Vec::new(); out_len];
        let mut indices = vec![Vec::new(); out_len];

        for out_idx in 0..out_len {
            let center = (out_idx as f32 + 0.5) * scale - 0.5;
            let start = (center - radius).floor() as isize;
            let end = (center + radius).ceil() as isize;
            let mut idxs = Vec::new();
            let mut wts = Vec::new();
            for src_idx in start..=end {
                let clamped = src_idx.clamp(0, (in_len as isize) - 1) as usize;
                let distance = (center - src_idx as f32) / support_scale;
                let weight = cubic(a, distance) / support_scale;
                if weight != 0.0 {
                    idxs.push(clamped);
                    wts.push(weight);
                }
            }
            let sum: f32 = wts.iter().sum();
            if sum != 0.0 {
                for w in &mut wts {
                    *w /= sum;
                }
            }
            weights[out_idx] = wts;
            indices[out_idx] = idxs;
        }

        (weights, indices)
    }

    let input = input.contiguous()?.to_dtype(DType::F32)?;
    let (batch, channels, in_h, in_w) = input.shape().dims4()?;
    ensure!(batch == 1, "bicubic resize expects batch size 1");

    if in_h == out_h && in_w == out_w {
        return Ok(input.clone());
    }

    let scale_y = in_h as f32 / out_h as f32;
    let scale_x = in_w as f32 / out_w as f32;
    let support_y = scale_y.max(1.0);
    let support_x = scale_x.max(1.0);
    let a = -0.75f32;

    let (wy, iy) = compute_axis_weights(in_h, out_h, scale_y, support_y, a);
    let (wx, ix) = compute_axis_weights(in_w, out_w, scale_x, support_x, a);

    let flat = input.flatten_all()?.to_vec1::<f32>()?;
    let mut tmp = vec![0f32; channels * out_h * in_w];

    for ch in 0..channels {
        for oh in 0..out_h {
            // 先对纵向做卷积（保持 width 不变），与 PyTorch 的实现尽量保持顺序一致。
            let mut acc = vec![0f32; in_w];
            for (k, &src_y) in iy[oh].iter().enumerate() {
                let weight = wy[oh][k];
                let row_offset = ((ch * in_h + src_y) * in_w) as usize;
                for x in 0..in_w {
                    acc[x] += flat[row_offset + x] * weight;
                }
            }
            let dst_offset = (ch * out_h + oh) * in_w;
            tmp[dst_offset..dst_offset + in_w].copy_from_slice(&acc);
        }
    }

    let mut out = vec![0f32; channels * out_h * out_w];
    for ch in 0..channels {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut value = 0f32;
                for (k, &src_x) in ix[ow].iter().enumerate() {
                    // 横向卷积阶段读取上一步的 tmp，最终得到目标像素。
                    let weight = wx[ow][k];
                    let idx = (ch * out_h + oh) * in_w + src_x;
                    value += tmp[idx] * weight;
                }
                let out_idx = (ch * out_h + oh) * out_w + ow;
                out[out_idx] = value;
            }
        }
    }

    Tensor::from_vec(out, (1, channels, out_h, out_w), input.device()).map_err(Into::into)
}
fn compute_relative_bias(
    q: &Tensor,
    spatial: (usize, usize),
    rel_pos_h: Option<&Tensor>,
    rel_pos_w: Option<&Tensor>,
) -> Result<Tensor> {
    let (batch, heads, _seq, head_dim) = q.shape().dims4()?;
    let (q_h, q_w) = spatial;
    let k_h = q_h;
    let k_w = q_w;
    let rel_pos_h_vec = if let Some(t) = rel_pos_h {
        get_rel_pos_vec(q_h, k_h, t)?
    } else {
        vec![0f32; q_h * k_h * head_dim]
    };
    let rel_pos_w_vec = if let Some(t) = rel_pos_w {
        get_rel_pos_vec(q_w, k_w, t)?
    } else {
        vec![0f32; q_w * k_w * head_dim]
    };
    let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
    let seq_len = q_h * q_w;
    let k_len = k_h * k_w;
    let mut attn_bias = vec![0f32; batch * heads * seq_len * k_len];
    for b in 0..batch {
        for h in 0..heads {
            let bh_index = b * heads + h;
            let mut rel_h_bias = vec![0f32; seq_len * k_h];
            let mut rel_w_bias = vec![0f32; seq_len * k_w];
            for qhi in 0..q_h {
                for qwi in 0..q_w {
                    let q_idx = qhi * q_w + qwi;
                    let q_offset = (bh_index * seq_len + q_idx) * head_dim;
                    for kh in 0..k_h {
                        let rel_offset = (qhi * k_h + kh) * head_dim;
                        let mut sum = 0f32;
                        for d in 0..head_dim {
                            sum += q_vec[q_offset + d] * rel_pos_h_vec[rel_offset + d];
                        }
                        rel_h_bias[q_idx * k_h + kh] = sum;
                    }
                    for kw in 0..k_w {
                        let rel_offset = (qwi * k_w + kw) * head_dim;
                        let mut sum = 0f32;
                        for d in 0..head_dim {
                            sum += q_vec[q_offset + d] * rel_pos_w_vec[rel_offset + d];
                        }
                        rel_w_bias[q_idx * k_w + kw] = sum;
                    }
                }
            }
            for q_idx in 0..seq_len {
                for kh in 0..k_h {
                    for kw in 0..k_w {
                        let col = kh * k_w + kw;
                        let bias_idx = ((bh_index * seq_len + q_idx) * k_len) + col;
                        attn_bias[bias_idx] =
                            rel_h_bias[q_idx * k_h + kh] + rel_w_bias[q_idx * k_w + kw];
                    }
                }
            }
        }
    }
    Ok(Tensor::from_vec(
        attn_bias,
        (batch, heads, seq_len, k_len),
        q.device(),
    )?)
}

fn get_rel_pos_vec(q_size: usize, k_size: usize, rel_pos: &Tensor) -> Result<Vec<f32>> {
    let (orig_len, head_dim) = rel_pos.shape().dims2()?;
    let rel_data = rel_pos.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let max_rel_dist = 2 * usize::max(q_size, k_size) - 1;
    let resized = if orig_len == max_rel_dist {
        rel_data
    } else {
        let mut resized = vec![vec![0f32; head_dim]; max_rel_dist];
        let scale = if max_rel_dist == 1 {
            0.0
        } else {
            (orig_len - 1) as f32 / (max_rel_dist - 1) as f32
        };
        for i in 0..max_rel_dist {
            let src_pos = scale * i as f32;
            let left = src_pos.floor() as usize;
            let right = (left + 1).min(orig_len - 1);
            let weight = src_pos - left as f32;
            for d in 0..head_dim {
                let left_val = rel_data[left][d];
                let right_val = rel_data[right][d];
                resized[i][d] = left_val * (1.0 - weight) + right_val * weight;
            }
        }
        resized
    };
    let scale_q = (k_size as f32 / q_size as f32).max(1.0);
    let scale_k = (q_size as f32 / k_size as f32).max(1.0);
    let mut output = vec![0f32; q_size * k_size * head_dim];
    for qi in 0..q_size {
        for ki in 0..k_size {
            let q_coord = qi as f32 * scale_q;
            let k_coord = ki as f32 * scale_k;
            let rel = (q_coord - k_coord) + (k_size as f32 - 1.0) * scale_k;
            let idx = rel.floor().clamp(0.0, (max_rel_dist - 1) as f32) as usize;
            let dest = (qi * k_size + ki) * head_dim;
            output[dest..dest + head_dim].copy_from_slice(&resized[idx][..]);
        }
    }
    Ok(output)
}

pub fn window_unpartition_shape(partition: WindowPartitionShape, window: usize) -> (usize, usize) {
    let height = partition.padded_height.min(partition.tiles_h * window);
    let width = partition.padded_width.min(partition.tiles_w * window);
    (height, width)
}
