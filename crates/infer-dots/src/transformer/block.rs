use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor, shape::D};
use candle_nn::{
    VarBuilder,
    ops::{rms_norm, softmax},
};
use deepseek_ocr_core::attention::{
    collect_cache_chunk_views, output_from_cache_chunks, scores_from_cache_chunks,
};
use deepseek_ocr_core::cache::{KvCacheChunk, KvCacheEntry};
use deepseek_ocr_core::tensor::{into_dtype_if_needed, to_dtype_if_needed};

use crate::{config::DotsOcrTextConfig, quant::QuantLinear, snapshot::SnapshotLinearMap};

#[derive(Debug)]
pub struct Qwen2Block {
    norm1: Tensor,
    norm2: Tensor,
    attention: Qwen2Attention,
    mlp: Qwen2Mlp,
    eps: f64,
}

impl Qwen2Block {
    pub fn load(
        cfg: &DotsOcrTextConfig,
        vb: &VarBuilder,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let norm1 = vb
            .pp("input_layernorm")
            .get(cfg.hidden_size, "weight")
            .context("missing input_layernorm weight")?;
        let norm2 = vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")
            .context("missing post_attention_layernorm weight")?;
        let mut snapshot_hits = snapshot_hits;
        let attn = Qwen2Attention::load(
            cfg,
            &vb.pp("self_attn"),
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        let mlp = Qwen2Mlp::load(cfg, &vb.pp("mlp"), snapshot_hits, snapshot_label)?;
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
    q_proj: QuantLinear,
    k_proj: QuantLinear,
    v_proj: QuantLinear,
    o_proj: QuantLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
}

impl Qwen2Attention {
    fn load(
        cfg: &DotsOcrTextConfig,
        vb: &VarBuilder,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let bias = cfg.attention_bias;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let kv_dim = cfg.num_key_value_heads.max(1) * head_dim;
        let mut snapshot_hits = snapshot_hits;
        let mut make_linear = |name: &str, out: usize| -> Result<QuantLinear> {
            let sub = vb.pp(name);
            let has_bias = bias && sub.contains_tensor("bias");
            QuantLinear::load(
                sub,
                out,
                cfg.hidden_size,
                has_bias,
                snapshot_hits.as_deref_mut(),
                snapshot_label,
            )
        };
        let q_proj = make_linear("q_proj", cfg.hidden_size)?;
        let k_proj = make_linear("k_proj", kv_dim)?;
        let v_proj = make_linear("v_proj", kv_dim)?;
        let o_proj = make_linear("o_proj", cfg.hidden_size)?;
        ensure!(
            cfg.hidden_size.is_multiple_of(cfg.num_attention_heads),
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
        let k = self
            .k_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = self
            .v_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let (q, k) = apply_rope(&q, &k, cos, sin, self.rope_dim)?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let chunk = KvCacheChunk::new(k_t.clone(), v.clone())?;
        let present = if use_cache { Some(chunk) } else { None };

        let repeats = self.num_heads / self.num_kv_heads;
        let cache_chunks = if let Some(entry) = past {
            Some(collect_cache_chunk_views(
                entry,
                batch,
                self.num_kv_heads,
                self.head_dim,
                self.head_dim,
            )?)
        } else {
            None
        };
        let past_len = cache_chunks.as_ref().map_or(0, |chunks| chunks.total_len);
        let total_len = past_len + seq_len;

        let q = make_contiguous(q.contiguous()?, force_contig)?;
        let q_dtype = q.dtype();
        let compute_dtype = compute_dtype_for(&q);
        let q = maybe_cast(&q, compute_dtype)?;

        let k_new = make_contiguous(repeat_kv(&k, repeats)?.transpose(2, 3)?, force_contig)?;
        let k_new = maybe_cast(&k_new, compute_dtype)?;
        let v_new = make_contiguous(repeat_kv(&v, repeats)?, force_contig)?;
        let v_new = maybe_cast(&v_new, compute_dtype)?;

        let mut expanded_cache_keys = Vec::new();
        let mut expanded_cache_values = Vec::new();
        if let Some(cache) = cache_chunks.as_ref() {
            expanded_cache_keys.reserve(cache.keys.len());
            expanded_cache_values.reserve(cache.values.len());
            for key_t in &cache.keys {
                let key_seq = key_t.transpose(2, 3)?;
                let key_t = make_contiguous(repeat_kv(&key_seq, repeats)?.transpose(2, 3)?, force_contig)?;
                expanded_cache_keys.push(maybe_cast(&key_t, compute_dtype)?);
            }
            for value in &cache.values {
                let value = make_contiguous(repeat_kv(value, repeats)?, force_contig)?;
                expanded_cache_values.push(maybe_cast(&value, compute_dtype)?);
            }
        }

        let mut scores = if expanded_cache_keys.is_empty() {
            q.matmul(&k_new)?
        } else {
            scores_from_cache_chunks(&q, &expanded_cache_keys, &k_new)?
        };
        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        let scale_tensor =
            Tensor::full(scale as f32, (), scores.device())?.to_dtype(compute_dtype)?;
        scores = scores.broadcast_mul(&scale_tensor)?;
        if let Some(mask) = attention_mask {
            let expanded = mask
                .expand((batch, self.num_heads, seq_len, total_len))?
                .contiguous()?;
            let expanded = maybe_cast(&expanded, compute_dtype)?;
            scores = scores.add(&expanded)?;
        }
        let probs = softmax(&scores, D::Minus1)?;

        let ctx = if expanded_cache_values.is_empty() {
            probs.matmul(&v_new)?
        } else {
            output_from_cache_chunks(&probs, &expanded_cache_values, &v_new, past_len, seq_len)?
        };
        let ctx = into_dtype_if_needed(ctx, q_dtype)?;
        let ctx = ctx
            .contiguous()?
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        let output = self.o_proj.forward(&ctx)?;
        Ok((output, present))
    }
}

#[derive(Debug)]
struct Qwen2Mlp {
    gate: QuantLinear,
    up: QuantLinear,
    down: QuantLinear,
}

impl Qwen2Mlp {
    fn load(
        cfg: &DotsOcrTextConfig,
        vb: &VarBuilder,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let mut snapshot_hits = snapshot_hits;
        let mut make_linear = |name: &str, input: usize, output: usize| -> Result<QuantLinear> {
            let sub = vb.pp(name);
            // Qwen2 MLPs use biases when present in the checkpoint; we rely on
            // VarBuilder to decide whether a bias tensor exists.
            let has_bias = sub.contains_tensor("bias");
            QuantLinear::load(
                sub,
                output,
                input,
                has_bias,
                snapshot_hits.as_deref_mut(),
                snapshot_label,
            )
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
        self.down.forward(&hidden)
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
    to_dtype_if_needed(tensor, dtype)
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
    let cos = to_dtype_if_needed(cos, rot_part.dtype())?;
    let sin = to_dtype_if_needed(sin, rot_part.dtype())?;
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
