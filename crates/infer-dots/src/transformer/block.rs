use anyhow::{Context, Result, ensure};
use candle_core::{DType, Module, Tensor, shape::D};
use candle_nn::{
    Linear, VarBuilder, linear, linear_no_bias,
    ops::{rms_norm, softmax},
};
use deepseek_ocr_core::cache::{KvCacheChunk, KvCacheEntry};

use crate::config::DotsOcrTextConfig;

#[derive(Debug)]
pub struct Qwen2Block {
    norm1: Tensor,
    norm2: Tensor,
    attention: Qwen2Attention,
    mlp: Qwen2Mlp,
    eps: f64,
}

impl Qwen2Block {
    pub fn load(cfg: &DotsOcrTextConfig, vb: &VarBuilder) -> Result<Self> {
        let norm1 = vb
            .pp("input_layernorm")
            .get(cfg.hidden_size, "weight")
            .context("missing input_layernorm weight")?;
        let norm2 = vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")
            .context("missing post_attention_layernorm weight")?;
        let attn = Qwen2Attention::load(cfg, &vb.pp("self_attn"))?;
        let mlp = Qwen2Mlp::load(cfg, &vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            norm2,
            attention: attn,
            mlp,
            eps: cfg.rms_norm_eps,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        past: Option<&KvCacheEntry>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<KvCacheChunk>)> {
        let normed = rms_norm(hidden_states, &self.norm1, self.eps as f32)
            .context("attention rms norm failed")?;
        let (attn_out, present) =
            self.attention
                .forward(&normed, cos, sin, attention_mask, past, use_cache)?;
        let residual = hidden_states.add(&attn_out)?;
        let normed =
            rms_norm(&residual, &self.norm2, self.eps as f32).context("mlp rms norm failed")?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok((residual.add(&mlp_out)?, present))
    }
}

#[derive(Debug)]
struct Qwen2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
}

impl Qwen2Attention {
    fn load(cfg: &DotsOcrTextConfig, vb: &VarBuilder) -> Result<Self> {
        let bias = cfg.attention_bias;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let kv_dim = cfg.num_key_value_heads.max(1) * head_dim;
        let make_linear = |name: &str, out: usize| -> Result<Linear> {
            let sub = vb.pp(name);
            let has_bias = bias && sub.contains_tensor("bias");
            if has_bias {
                Ok(linear(cfg.hidden_size, out, sub)?)
            } else {
                Ok(linear_no_bias(cfg.hidden_size, out, sub)?)
            }
        };
        let q_proj = make_linear("q_proj", cfg.hidden_size)?;
        let k_proj = make_linear("k_proj", kv_dim)?;
        let v_proj = make_linear("v_proj", kv_dim)?;
        let o_proj = make_linear("o_proj", cfg.hidden_size)?;
        ensure!(
            cfg.hidden_size % cfg.num_attention_heads == 0,
            "hidden_size {} not divisible by num_attention_heads {}",
            cfg.hidden_size,
            cfg.num_attention_heads
        );
        let num_kv_heads = cfg.num_key_value_heads.max(1);
        let rope_dim = head_dim;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        past: Option<&KvCacheEntry>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<KvCacheChunk>)> {
        let (batch, seq_len, hidden) = hidden_states.shape().dims3()?;
        let force_contig = hidden_states.device().is_cpu();
        ensure!(
            hidden == self.num_heads * self.head_dim,
            "hidden dim mismatch: got {}, expected {}",
            hidden,
            self.num_heads * self.head_dim
        );
        let q = self
            .q_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let mut k = self
            .k_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = self
            .v_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let (q, k_rot) = apply_rope(&q, &k, cos, sin, self.rope_dim)?;
        k = k_rot;
        let k_t = k.transpose(2, 3)?;
        let chunk = KvCacheChunk::new(k_t.clone(), v.clone())?;
        let mut key_total = chunk.key_t.clone();
        let mut value_total = chunk.value.clone();
        if let Some(entry) = past {
            let past_k = entry.key_view()?;
            let past_v = entry.value_view()?;
            key_total = Tensor::cat(&[past_k, key_total], 3)?;
            value_total = Tensor::cat(&[past_v, value_total], 2)?;
        }
        let present = if use_cache { Some(chunk) } else { None };
        let k_seq = key_total.transpose(2, 3)?;
        let k_seq = repeat_kv(&k_seq, self.num_heads / self.num_kv_heads)?;
        let v_full = repeat_kv(&value_total, self.num_heads / self.num_kv_heads)?;
        let (_, _, total_len, _) = k_seq.shape().dims4()?;
        let q_flat = make_contiguous(
            q.reshape((batch * self.num_heads, seq_len, self.head_dim))?,
            force_contig,
        )?;
        // Accelerate backend (via `metal`/`accelerate` features) requires contiguous
        // RHS tensors in batched matmuls, so only force when running on CPU/Accelerate.
        let k_tiled = make_contiguous(
            k_seq
                .transpose(2, 3)?
                .reshape((batch * self.num_heads, self.head_dim, total_len))?,
            force_contig,
        )?;
        let compute_dtype = compute_dtype_for(&q_flat);
        let mut scores =
            maybe_cast(&q_flat, compute_dtype)?.matmul(&maybe_cast(&k_tiled, compute_dtype)?)?;
        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        let scale_tensor =
            Tensor::full(scale as f32, (), scores.device())?.to_dtype(compute_dtype)?;
        scores = scores.broadcast_mul(&scale_tensor)?;
        if let Some(mask) = attention_mask {
            let expanded = mask
                .expand((batch, self.num_heads, seq_len, total_len))?
                .reshape((batch * self.num_heads, seq_len, total_len))?;
            let expanded = maybe_cast(&expanded, compute_dtype)?;
            scores = scores.add(&expanded)?;
        }
        let probs = softmax(&scores, D::Minus1)?;
        let v_flat = make_contiguous(
            v_full.reshape((batch * self.num_heads, total_len, self.head_dim))?,
            force_contig,
        )?;
        let mut ctx = probs.matmul(&maybe_cast(&v_flat, compute_dtype)?)?;
        if compute_dtype != q_flat.dtype() {
            ctx = ctx.to_dtype(q_flat.dtype())?;
        }
        let ctx = ctx
            .reshape((batch, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        let output = self.o_proj.forward(&ctx)?;
        Ok((output, present))
    }
}

#[derive(Debug)]
struct Qwen2Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Qwen2Mlp {
    fn load(cfg: &DotsOcrTextConfig, vb: &VarBuilder) -> Result<Self> {
        let make_linear = |name: &str, input: usize, output: usize| -> Result<Linear> {
            let sub = vb.pp(name);
            if sub.contains_tensor("bias") {
                Ok(linear(input, output, sub)?)
            } else {
                Ok(linear_no_bias(input, output, sub)?)
            }
        };
        let gate = make_linear("gate_proj", cfg.hidden_size, cfg.intermediate_size)?;
        let up = make_linear("up_proj", cfg.hidden_size, cfg.intermediate_size)?;
        let down = make_linear("down_proj", cfg.intermediate_size, cfg.hidden_size)?;
        Ok(Self { gate, up, down })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(input)?.silu()?;
        let up = self.up.forward(input)?;
        let hidden = gate.broadcast_mul(&up)?;
        Ok(self.down.forward(&hidden)?)
    }
}

fn repeat_kv(t: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(t.clone());
    }
    let (batch, heads, seq, dim) = t.shape().dims4()?;
    let expanded = t
        .unsqueeze(2)?
        .expand((batch, heads, repeats, seq, dim))?
        .reshape((batch, heads * repeats, seq, dim))?;
    Ok(expanded)
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

fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rope_dim: usize,
) -> Result<(Tensor, Tensor)> {
    if rope_dim == 0 {
        return Ok((q.clone(), k.clone()));
    }
    let (batch, q_heads, seq_len, _) = q.shape().dims4()?;
    let (_, k_heads, _, _) = k.shape().dims4()?;
    let cos_q = cos
        .expand((batch, q_heads, seq_len, rope_dim))?
        .contiguous()?;
    let sin_q = sin
        .expand((batch, q_heads, seq_len, rope_dim))?
        .contiguous()?;
    let q_rot = apply_rotary_inner(q, &cos_q, &sin_q, rope_dim)?;
    let cos_k = cos
        .expand((batch, k_heads, seq_len, rope_dim))?
        .contiguous()?;
    let sin_k = sin
        .expand((batch, k_heads, seq_len, rope_dim))?
        .contiguous()?;
    let k_rot = apply_rotary_inner(k, &cos_k, &sin_k, rope_dim)?;
    Ok((q_rot, k_rot))
}

fn apply_rotary_inner(t: &Tensor, cos: &Tensor, sin: &Tensor, rope_dim: usize) -> Result<Tensor> {
    let head_dim = t.dim(D::Minus1)?;
    ensure!(
        rope_dim <= head_dim,
        "rope dimension {} exceeds head dim {}",
        rope_dim,
        head_dim
    );
    let (rot_part, pass_part) = if rope_dim == head_dim {
        (t.clone(), None)
    } else {
        let rot = t.narrow(D::Minus1, 0, rope_dim)?;
        let pass = t.narrow(D::Minus1, rope_dim, head_dim - rope_dim)?;
        (rot, Some(pass))
    };
    let rotated = rotate_half(&rot_part)?;
    let cos = if cos.dtype() == rot_part.dtype() {
        cos.clone()
    } else {
        cos.to_dtype(rot_part.dtype())?
    };
    let sin = if sin.dtype() == rot_part.dtype() {
        sin.clone()
    } else {
        sin.to_dtype(rot_part.dtype())?
    };
    let rot = rot_part
        .broadcast_mul(&cos)?
        .add(&rotated.broadcast_mul(&sin)?)?;
    if let Some(pass) = pass_part {
        Ok(Tensor::cat(&[rot, pass], D::Minus1)?)
    } else {
        Ok(rot)
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    ensure!(last % 2 == 0, "rotate_half expects even dim, got {last}");
    let left = x.narrow(D::Minus1, 0, last / 2)?;
    let right = x.narrow(D::Minus1, last / 2, last / 2)?;
    let neg_right = right.neg()?;
    Ok(Tensor::cat(&[neg_right, left], D::Minus1)?)
}
