use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Module, Tensor, shape::D};
use candle_nn::{
    Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder, conv2d, conv2d_no_bias, layer_norm,
    linear, linear_no_bias,
    ops::{rms_norm, softmax},
};

use crate::config::DotsVisionConfig;

#[allow(dead_code)]
#[derive(Debug)]
pub struct DotsVisionModel {
    config: Arc<DotsVisionConfig>,
    patch_embed: DotsPatchEmbed,
    blocks: Vec<DotsVisionBlock>,
    merger: PatchMerger,
    post_norm: Option<Tensor>,
    rotary: VisionRotaryEmbedding,
    device: Device,
    dtype: DType,
}

impl DotsVisionModel {
    pub fn load(config: Arc<DotsVisionConfig>, vb: &VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let patch_embed = DotsPatchEmbed::load(config.as_ref(), &vb.pp("patch_embed"))?;
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            let block_vb = vb.pp(&format!("blocks.{idx}"));
            blocks.push(DotsVisionBlock::load(config.as_ref(), &block_vb)?);
        }
        let post_norm = if config.post_norm {
            Some(
                vb.pp("post_trunk_norm")
                    .get(config.embed_dim, "weight")
                    .context("missing post_trunk_norm weight")?,
            )
        } else {
            None
        };
        let merger = PatchMerger::load(config.as_ref(), &vb.pp("merger"))?;
        let rotary = VisionRotaryEmbedding::new(config.as_ref(), &device)?;
        Ok(Self {
            config,
            patch_embed,
            blocks,
            merger,
            post_norm,
            rotary,
            device,
            dtype,
        })
    }

    pub fn forward(&self, pixel_values: &Tensor, grid_thw: &[[u32; 3]]) -> Result<Tensor> {
        let patch_tokens = self.patch_embed.forward(pixel_values)?;
        let layout = SequenceLayout::from_grid(grid_thw, self.config.spatial_merge_size)?;
        let (token_count, _) = patch_tokens.dims2()?;
        ensure!(
            token_count == layout.total_tokens,
            "patch token count mismatch"
        );
        let rotary = self.rotary.build_embeddings(&layout)?;
        let mut hidden = patch_tokens;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &layout, &rotary)?;
        }
        if let Some(post) = &self.post_norm {
            hidden = rms_norm(&hidden, post, self.config.rms_norm_eps as f32)
                .context("post trunk rms_norm failed")?;
        }
        self.merger.forward(&hidden, &layout)
    }
}

#[derive(Debug, Clone)]
struct FrameLayout {
    start: usize,
    len: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SequenceLayout {
    frames: Vec<FrameLayout>,
    total_tokens: usize,
    merge_groups: usize,
    group_size: usize,
    positions: Vec<[u32; 2]>,
}

impl SequenceLayout {
    fn from_grid(grid_thw: &[[u32; 3]], merge: usize) -> Result<Self> {
        let mut total = 0usize;
        let mut groups = 0usize;
        let mut frames = Vec::new();
        let mut positions = Vec::new();
        let group_size = merge * merge;
        for &[t, h, w] in grid_thw {
            ensure!(
                h as usize % merge == 0 && w as usize % merge == 0,
                "grid dims {}x{} not divisible by merge {}",
                h,
                w,
                merge
            );
            let h = h as usize;
            let w = w as usize;
            let patches_per_frame = h * w;
            ensure!(
                patches_per_frame % group_size == 0,
                "patch grid {} not divisible by merge group {}",
                patches_per_frame,
                group_size
            );
            let frame_positions = build_frame_positions(h, w, merge)?;
            ensure!(
                frame_positions.len() == patches_per_frame,
                "frame positions mismatch {} vs {}",
                frame_positions.len(),
                patches_per_frame
            );
            let frames_per_image = t as usize;
            for _ in 0..frames_per_image {
                let start = total;
                total += patches_per_frame;
                groups += patches_per_frame / group_size;
                frames.push(FrameLayout {
                    start,
                    len: patches_per_frame,
                });
                positions.extend_from_slice(&frame_positions);
            }
        }
        ensure!(
            positions.len() == total,
            "layout position count {} mismatches total tokens {}",
            positions.len(),
            total
        );
        Ok(Self {
            frames,
            total_tokens: total,
            merge_groups: groups,
            group_size,
            positions,
        })
    }

    fn frames(&self) -> &[FrameLayout] {
        &self.frames
    }

    fn positions(&self) -> &[[u32; 2]] {
        &self.positions
    }

    fn uniform_frame_len(&self) -> Option<usize> {
        let mut iter = self.frames.iter().filter(|frame| frame.len > 0);
        let first = iter.next()?.len;
        if iter.all(|frame| frame.len == first) {
            Some(first)
        } else {
            None
        }
    }
}

fn build_frame_positions(height: usize, width: usize, merge: usize) -> Result<Vec<[u32; 2]>> {
    let mut positions = Vec::with_capacity(height * width);
    ensure!(
        height % merge == 0 && width % merge == 0,
        "frame {}x{} incompatible with merge {}",
        height,
        width,
        merge
    );
    let blocks_h = height / merge;
    let blocks_w = width / merge;
    for bh in 0..blocks_h {
        for bw in 0..blocks_w {
            for ih in 0..merge {
                for iw in 0..merge {
                    let hpos = bh * merge + ih;
                    let wpos = bw * merge + iw;
                    positions.push([hpos as u32, wpos as u32]);
                }
            }
        }
    }
    Ok(positions)
}

#[derive(Debug)]
struct DotsPatchEmbed {
    proj: Conv2d,
    norm_weight: Tensor,
    eps: f64,
}

impl DotsPatchEmbed {
    fn load(cfg: &DotsVisionConfig, vb: &VarBuilder) -> Result<Self> {
        let mut conv_cfg = Conv2dConfig::default();
        conv_cfg.stride = cfg.patch_size;
        conv_cfg.padding = 0;
        let proj = if vb.contains_tensor("patchifier.proj.bias") {
            conv2d(
                cfg.num_channels,
                cfg.embed_dim,
                cfg.patch_size,
                conv_cfg,
                vb.pp("patchifier").pp("proj"),
            )?
        } else {
            conv2d_no_bias(
                cfg.num_channels,
                cfg.embed_dim,
                cfg.patch_size,
                conv_cfg,
                vb.pp("patchifier").pp("proj"),
            )?
        };
        let norm_weight = vb
            .pp("patchifier")
            .pp("norm")
            .get(cfg.embed_dim, "weight")
            .context("missing patchifier norm weight")?;
        Ok(Self {
            proj,
            norm_weight,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let projected = self.proj.forward(input)?;
        let (batch, embed_dim, _, _) = projected.dims4()?;
        let reshaped = projected.reshape((batch, embed_dim))?;
        rms_norm(&reshaped, &self.norm_weight, self.eps as f32).context("patch rms norm failed")
    }
}

#[derive(Debug)]
struct DotsVisionBlock {
    norm1: Tensor,
    norm2: Tensor,
    attn: VisionAttention,
    mlp: DotsSwiGLUFFN,
    eps: f64,
}

impl DotsVisionBlock {
    fn load(cfg: &DotsVisionConfig, vb: &VarBuilder) -> Result<Self> {
        let norm1 = vb
            .pp("norm1")
            .get(cfg.embed_dim, "weight")
            .context("missing norm1 weight")?;
        let norm2 = vb
            .pp("norm2")
            .get(cfg.embed_dim, "weight")
            .context("missing norm2 weight")?;
        let attn = VisionAttention::load(cfg, &vb.pp("attn"))?;
        let mlp = DotsSwiGLUFFN::load(cfg, &vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, hidden: &Tensor, layout: &SequenceLayout, rotary: &Tensor) -> Result<Tensor> {
        let normed =
            rms_norm(hidden, &self.norm1, self.eps as f32).context("vision block norm1 failed")?;
        let attn_out = self.attn.forward(&normed, layout, rotary)?;
        let residual = hidden.add(&attn_out)?;
        let normed2 = rms_norm(&residual, &self.norm2, self.eps as f32)
            .context("vision block norm2 failed")?;
        let mlp_out = self.mlp.forward(&normed2)?;
        Ok(residual.add(&mlp_out)?)
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn load(cfg: &DotsVisionConfig, vb: &VarBuilder) -> Result<Self> {
        let qkv_vb = vb.pp("qkv");
        let proj_vb = vb.pp("proj");
        let qkv = if cfg.use_bias {
            linear(cfg.embed_dim, cfg.embed_dim * 3, qkv_vb)?
        } else {
            linear_no_bias(cfg.embed_dim, cfg.embed_dim * 3, qkv_vb)?
        };
        let proj = if cfg.use_bias {
            linear(cfg.embed_dim, cfg.embed_dim, proj_vb)?
        } else {
            linear_no_bias(cfg.embed_dim, cfg.embed_dim, proj_vb)?
        };
        ensure!(
            cfg.embed_dim % cfg.num_attention_heads == 0,
            "embed_dim not divisible by num_heads"
        );
        Ok(Self {
            qkv,
            proj,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.embed_dim / cfg.num_attention_heads,
        })
    }

    fn forward(&self, hidden: &Tensor, layout: &SequenceLayout, rotary: &Tensor) -> Result<Tensor> {
        let qkv = self.qkv.forward(hidden)?;
        let total = qkv.dim(0)?;
        ensure!(
            qkv.dim(1)? == self.num_heads * self.head_dim * 3,
            "unexpected qkv projection size"
        );
        let reshaped = qkv.reshape((total, 3, self.num_heads, self.head_dim))?;
        let q = reshaped
            .narrow(1, 0, 1)?
            .reshape((total, self.num_heads, self.head_dim))?;
        let k = reshaped
            .narrow(1, 1, 1)?
            .reshape((total, self.num_heads, self.head_dim))?;
        let v = reshaped
            .narrow(1, 2, 1)?
            .reshape((total, self.num_heads, self.head_dim))?;

        let (q, k) = apply_rotary(&q, &k, rotary)?;
        if let Some(frame_len) = layout.uniform_frame_len() {
            return self.forward_uniform(&q, &k, &v, layout, frame_len);
        }
        let mut outputs = Vec::with_capacity(layout.frames().len());
        for frame in layout.frames() {
            let start = frame.start;
            let len = frame.len;
            if len == 0 {
                continue;
            }
            let q_seq = q.narrow(0, start, len)?;
            let k_seq = k.narrow(0, start, len)?;
            let v_seq = v.narrow(0, start, len)?;
            let force_contig = q_seq.device().is_cpu();
            let q_heads = make_contiguous(q_seq.transpose(0, 1)?, force_contig)?;
            let k_heads = make_contiguous(k_seq.transpose(0, 1)?, force_contig)?;
            let v_heads = make_contiguous(v_seq.transpose(0, 1)?, force_contig)?;
            let k_t = make_contiguous(k_heads.transpose(1, 2)?, force_contig)?;
            let compute_dtype = compute_dtype_for(&q_heads);
            let attn_scores =
                maybe_cast(&q_heads, compute_dtype)?.matmul(&maybe_cast(&k_t, compute_dtype)?)?;
            let scale = 1.0f64 / (self.head_dim as f64).sqrt();
            let scale_tensor =
                Tensor::full(scale as f32, (), attn_scores.device())?.to_dtype(compute_dtype)?;
            let attn_scores = attn_scores.broadcast_mul(&scale_tensor)?;
            let probs = softmax(&attn_scores, D::Minus1)?;
            let mut ctx = probs.matmul(&maybe_cast(&v_heads, compute_dtype)?)?;
            if compute_dtype != q_heads.dtype() {
                ctx = ctx.to_dtype(q_heads.dtype())?;
            }
            let ctx = ctx
                .transpose(0, 1)?
                .reshape((len, self.num_heads * self.head_dim))?;
            outputs.push(ctx);
        }
        let concatenated = match outputs.len() {
            0 => Tensor::zeros(
                (0, self.num_heads * self.head_dim),
                hidden.dtype(),
                hidden.device(),
            )?,
            1 => outputs.into_iter().next().unwrap(),
            _ => {
                let refs: Vec<&Tensor> = outputs.iter().collect();
                Tensor::cat(&refs, 0)?
            }
        };
        Ok(self.proj.forward(&concatenated)?)
    }

    fn forward_uniform(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        layout: &SequenceLayout,
        frame_len: usize,
    ) -> Result<Tensor> {
        let frame_count = layout.frames().len();
        if frame_count == 0 || frame_len == 0 {
            return Ok(Tensor::zeros(
                (0, self.num_heads * self.head_dim),
                q.dtype(),
                q.device(),
            )?);
        }
        let total = frame_count * frame_len;
        ensure!(
            q.dim(0)? == total && k.dim(0)? == total && v.dim(0)? == total,
            "uniform layout mismatch: total tokens {}, frame {}×{}",
            total,
            frame_count,
            frame_len
        );
        let force_contig = q.device().is_cpu();
        let q_heads = make_contiguous(
            q.contiguous()?
                .reshape((frame_count, frame_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .reshape((frame_count * self.num_heads, frame_len, self.head_dim))?,
            force_contig,
        )?;
        let k_heads = make_contiguous(
            k.contiguous()?
                .reshape((frame_count, frame_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .reshape((frame_count * self.num_heads, frame_len, self.head_dim))?,
            force_contig,
        )?;
        let v_heads = make_contiguous(
            v.contiguous()?
                .reshape((frame_count, frame_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .reshape((frame_count * self.num_heads, frame_len, self.head_dim))?,
            force_contig,
        )?;
        let k_t = make_contiguous(k_heads.transpose(1, 2)?, force_contig)?;
        let compute_dtype = compute_dtype_for(&q_heads);
        let attn_scores =
            maybe_cast(&q_heads, compute_dtype)?.matmul(&maybe_cast(&k_t, compute_dtype)?)?;
        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        let scale_tensor =
            Tensor::full(scale as f32, (), attn_scores.device())?.to_dtype(compute_dtype)?;
        let attn_scores = attn_scores.broadcast_mul(&scale_tensor)?;
        let probs = softmax(&attn_scores, D::Minus1)?;
        let mut ctx = probs.matmul(&maybe_cast(&v_heads, compute_dtype)?)?;
        if compute_dtype != q_heads.dtype() {
            ctx = ctx.to_dtype(q_heads.dtype())?;
        }
        let ctx = ctx
            .reshape((frame_count, self.num_heads, frame_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((frame_count * frame_len, self.num_heads * self.head_dim))?;
        Ok(self.proj.forward(&ctx)?)
    }
}

#[derive(Debug)]
struct DotsSwiGLUFFN {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

fn apply_rotary(q: &Tensor, k: &Tensor, rope: &Tensor) -> Result<(Tensor, Tensor)> {
    let len = q.dim(0)?;
    let heads = q.dim(1)?;
    let head_dim = q.dim(2)?;
    let rope_len = rope.dim(0)?;
    let rope_dim = rope.dim(1)?;
    ensure!(
        rope_len == len,
        "rope length {} must match token count {}",
        rope_len,
        len
    );
    if len == 0 || rope_dim == 0 {
        return Ok((q.clone(), k.clone()));
    }
    ensure!(
        rope_dim * 2 == head_dim,
        "rope dim {} incompatible with head dim {}",
        rope_dim,
        head_dim
    );
    let rope = if rope.dtype() == DType::F32 {
        rope.clone()
    } else {
        rope.to_dtype(DType::F32)?
    };
    let cos = rope.cos()?.unsqueeze(1)?;
    let cos = Tensor::cat(&[cos.clone(), cos], 2)?
        .expand((len, heads, head_dim))?
        .contiguous()?;
    let sin = rope.sin()?.unsqueeze(1)?;
    let sin = Tensor::cat(&[sin.clone(), sin], 2)?
        .expand((len, heads, head_dim))?
        .contiguous()?;
    let q_base = if q.dtype() == DType::F32 {
        q.clone()
    } else {
        q.to_dtype(DType::F32)?
    };
    let k_base = if k.dtype() == DType::F32 {
        k.clone()
    } else {
        k.to_dtype(DType::F32)?
    };
    let q_rot = apply_rotary_to(&q_base, &cos, &sin)?;
    let k_rot = apply_rotary_to(&k_base, &cos, &sin)?;
    Ok((q_rot.to_dtype(q.dtype())?, k_rot.to_dtype(k.dtype())?))
}

fn apply_rotary_to(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rotated = rotate_half(input)?;
    let left = input.broadcast_mul(cos)?;
    let right = rotated.broadcast_mul(sin)?;
    Ok(left.add(&right)?)
}

fn rotate_half(input: &Tensor) -> Result<Tensor> {
    let head_dim = input.dim(2)?;
    ensure!(
        head_dim % 2 == 0,
        "head dim {} must be even for rotary",
        head_dim
    );
    let half = head_dim / 2;
    let first = input.narrow(2, 0, half)?;
    let second = input.narrow(2, half, half)?;
    Ok(Tensor::cat(&[second.neg()?, first], 2)?)
}

fn make_contiguous(tensor: Tensor, force: bool) -> Result<Tensor> {
    if force {
        Ok(tensor.force_contiguous()?)
    } else {
        Ok(tensor.contiguous()?)
    }
}

fn compute_dtype_for(tensor: &Tensor) -> DType {
    match tensor.dtype() {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    }
}

fn maybe_cast(tensor: &Tensor, dtype: DType) -> Result<Tensor> {
    if tensor.dtype() == dtype {
        Ok(tensor.clone())
    } else {
        Ok(tensor.to_dtype(dtype)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_positions_follow_merge_groups() -> Result<()> {
        let layout = SequenceLayout::from_grid(&[[1, 4, 4]], 2)?;
        assert_eq!(layout.total_tokens, 16);
        assert_eq!(layout.merge_groups, 4);
        let expected = [
            [0u32, 0u32],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
        ];
        assert_eq!(&layout.positions()[..8], &expected);
        Ok(())
    }
}

impl DotsSwiGLUFFN {
    fn load(cfg: &DotsVisionConfig, vb: &VarBuilder) -> Result<Self> {
        let make_linear = |input: usize, output: usize, name: &str| -> Result<Linear> {
            let sub = vb.pp(name);
            if cfg.use_bias {
                Ok(linear(input, output, sub)?)
            } else {
                Ok(linear_no_bias(input, output, sub)?)
            }
        };
        let fc1 = make_linear(cfg.embed_dim, cfg.intermediate_size, "fc1")?;
        let fc2 = make_linear(cfg.intermediate_size, cfg.embed_dim, "fc2")?;
        let fc3 = make_linear(cfg.embed_dim, cfg.intermediate_size, "fc3")?;
        Ok(Self { fc1, fc2, fc3 })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = self.fc1.forward(input)?.silu()?;
        let up = self.fc3.forward(input)?;
        let hidden = gate.broadcast_mul(&up)?;
        Ok(self.fc2.forward(&hidden)?)
    }
}

#[derive(Debug)]
struct PatchMerger {
    ln_q: LayerNorm,
    mlp_in: Linear,
    mlp_out: Linear,
    merge_size: usize,
    embed_dim: usize,
}

impl PatchMerger {
    fn load(cfg: &DotsVisionConfig, vb: &VarBuilder) -> Result<Self> {
        let ln_q = layer_norm(cfg.embed_dim, 1e-6, vb.pp("ln_q"))?;
        let hidden = cfg.embed_dim * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let mlp_in = linear(hidden, hidden, vb.pp("mlp").pp("0"))?;
        let mlp_out = linear(hidden, cfg.hidden_size, vb.pp("mlp").pp("2"))?;
        Ok(Self {
            ln_q,
            mlp_in,
            mlp_out,
            merge_size: cfg.spatial_merge_size,
            embed_dim: cfg.embed_dim,
        })
    }

    fn forward(&self, hidden: &Tensor, layout: &SequenceLayout) -> Result<Tensor> {
        let normed = self.ln_q.forward(hidden)?;
        let reshaped = normed.reshape((
            layout.merge_groups,
            self.embed_dim * self.merge_size * self.merge_size,
        ))?;
        let pre = self.mlp_in.forward(&reshaped)?.gelu()?;
        let out = self.mlp_out.forward(&pre)?;
        Ok(out)
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct VisionRotaryEmbedding {
    rope_dim: usize,
    inv_freq: Vec<f32>,
    device: Device,
}

impl VisionRotaryEmbedding {
    fn new(cfg: &DotsVisionConfig, device: &Device) -> Result<Self> {
        let head_dim = cfg.embed_dim / cfg.num_attention_heads;
        ensure!(head_dim % 4 == 0, "vision head dim must be divisible by 4");
        let rope_dim = head_dim / 2;
        let axis_dim = rope_dim / 2;
        let mut inv_freq = Vec::with_capacity(axis_dim);
        for idx in 0..axis_dim {
            let exponent = (2 * idx) as f32 / (rope_dim as f32);
            inv_freq.push(1.0f32 / (10_000f32.powf(exponent)));
        }
        Ok(Self {
            rope_dim,
            inv_freq,
            device: device.clone(),
        })
    }

    fn build_embeddings(&self, layout: &SequenceLayout) -> Result<Tensor> {
        let mut data = Vec::with_capacity(layout.total_tokens * self.rope_dim);
        for &[hpos, wpos] in layout.positions() {
            let hpos = hpos as f32;
            let wpos = wpos as f32;
            for &freq in &self.inv_freq {
                data.push(hpos * freq);
            }
            for &freq in &self.inv_freq {
                data.push(wpos * freq);
            }
        }
        Ok(Tensor::from_vec(
            data,
            (layout.total_tokens, self.rope_dim),
            &self.device,
        )?)
    }
}
