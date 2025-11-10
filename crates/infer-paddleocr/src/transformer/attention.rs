use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, IndexOp, Tensor, shape::D};
#[cfg(feature = "flash-attn")]
use candle_flash_attn::flash_attn;
use candle_nn::ops::softmax;

use crate::config::PaddleOcrVlConfig;

use super::{
    cache::{KvCacheChunk, KvCacheEntry},
    ops::{apply_linear, rotate_half},
    rope::ErnieRotaryEmbedding,
    weights::ErnieAttentionWeights,
};

pub struct AttentionContext<'a> {
    pub cfg: &'a PaddleOcrVlConfig,
    pub rotary: &'a ErnieRotaryEmbedding,
}

pub fn supports_flash_attention(cfg: &PaddleOcrVlConfig, device: &Device, dtype: DType) -> bool {
    #[cfg(feature = "flash-attn")]
    {
        if !cfg.use_flash_attention {
            return false;
        }
        if !device.is_cuda() {
            return false;
        }
        matches!(dtype, DType::F16 | DType::BF16)
            && cfg.head_dim % 8 == 0
            && cfg.head_dim <= 256
            && cfg.hidden_size == cfg.num_attention_heads * cfg.head_dim
            && cfg.num_attention_heads % cfg.resolved_num_key_value_heads() == 0
    }
    #[cfg(not(feature = "flash-attn"))]
    {
        let _ = (cfg, device, dtype);
        false
    }
}

pub fn attention_forward(
    ctx: &AttentionContext<'_>,
    hidden_states: &Tensor,
    weights: &ErnieAttentionWeights,
    cos: &Tensor,
    sin: &Tensor,
    attn_bias: Option<&Tensor>,
    past_key_value: Option<&KvCacheEntry>,
    use_cache: bool,
) -> Result<(Tensor, Option<KvCacheChunk>)> {
    if ctx.cfg.use_flash_attention {
        if let Some(result) = flash_attention_forward(
            ctx,
            hidden_states,
            weights,
            cos,
            sin,
            attn_bias,
            past_key_value,
            use_cache,
        )? {
            return Ok(result);
        }
    }

    let cfg = ctx.cfg;
    let (batch, seq_len, hidden_size) = hidden_states.shape().dims3()?;
    ensure!(
        hidden_size == cfg.hidden_size,
        "hidden size mismatch: cfg {} tensor {}",
        cfg.hidden_size,
        hidden_size
    );
    let head_dim = cfg.head_dim;
    let num_heads = cfg.num_attention_heads;
    let num_kv_heads = cfg.resolved_num_key_value_heads();
    ensure!(
        num_heads % num_kv_heads == 0,
        "num_attention_heads {} must be divisible by num_key_value_heads {}",
        num_heads,
        num_kv_heads
    );
    let q = apply_linear(hidden_states, &weights.q_proj)?
        .reshape((batch, seq_len, num_heads, head_dim))?
        .permute((0, 2, 1, 3))?;
    let k = apply_linear(hidden_states, &weights.k_proj)?
        .reshape((batch, seq_len, num_kv_heads, head_dim))?
        .permute((0, 2, 1, 3))?;
    let v = apply_linear(hidden_states, &weights.v_proj)?
        .reshape((batch, seq_len, num_kv_heads, head_dim))?
        .permute((0, 2, 1, 3))?;

    let (q, k) = apply_multimodal_rotary(q, k, cos, sin, ctx.rotary.doubled_sections())?;

    let repeats = num_heads / num_kv_heads;
    let mut k = repeat_kv(&k, repeats)?;
    let mut v = repeat_kv(&v, repeats)?;

    let mut cache_key_t_view: Option<Tensor> = None;
    let mut cache_value_view: Option<Tensor> = None;
    let past_len = if let Some(entry) = past_key_value {
        let key_view = entry.key_view()?;
        let value_view = entry.value_view()?;
        validate_cache_shapes(cfg, &key_view, &value_view, batch, head_dim)?;
        cache_key_t_view = Some(key_view);
        cache_value_view = Some(value_view);
        entry.seq_len()
    } else {
        0
    };

    let q = q.contiguous()?;
    k = k.contiguous()?;
    v = v.contiguous()?;

    let k_t = k.permute((0, 1, 3, 2))?.contiguous()?;
    let attn_scores_new = q.matmul(&k_t)?;
    let attn_scores = if let Some(cache_key) = cache_key_t_view.as_ref() {
        if past_len > 0 {
            let cache_scores = q.matmul(&cache_key.contiguous()?)?;
            Tensor::cat(&[cache_scores, attn_scores_new], D::Minus1)?
        } else {
            attn_scores_new
        }
    } else {
        attn_scores_new
    };

    let scale = (head_dim as f64).sqrt();
    let mut attn_scores = (attn_scores / scale)?;
    if let Some(bias) = attn_bias {
        attn_scores = attn_scores.broadcast_add(bias)?;
    }
    let attn_weights = softmax(&attn_scores, D::Minus1).context("attention softmax failed")?;

    let attn_output = if let Some(cache_value) = cache_value_view.as_ref() {
        let cache_value = cache_value.contiguous()?;
        if past_len > 0 {
            let cached = attn_weights
                .narrow(D::Minus1, 0, past_len)?
                .matmul(&cache_value)?;
            let current = attn_weights
                .narrow(D::Minus1, past_len, seq_len)?
                .matmul(&v)?;
            cached.add(&current)?
        } else {
            attn_weights.matmul(&v)?
        }
    } else {
        attn_weights.matmul(&v)?
    };

    let present = if use_cache {
        Some(KvCacheChunk::new(k_t.clone(), v.clone())?)
    } else {
        None
    };

    let context =
        attn_output
            .permute((0, 2, 1, 3))?
            .reshape((batch, seq_len, num_heads * head_dim))?;

    let output = apply_linear(&context, &weights.o_proj)?;
    Ok((output, present))
}

fn repeat_kv(tensor: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(tensor.clone());
    }
    let (batch, heads, seq_len, head_dim) = tensor.shape().dims4()?;
    let expanded = tensor
        .unsqueeze(2)?
        .expand((batch, heads, repeats, seq_len, head_dim))?
        .reshape((batch, heads * repeats, seq_len, head_dim))?;
    Ok(expanded)
}

fn apply_multimodal_rotary(
    q: Tensor,
    k: Tensor,
    cos: &Tensor,
    sin: &Tensor,
    sections: &[usize],
) -> Result<(Tensor, Tensor)> {
    let cos_base = select_sections(cos, sections)?;
    let sin_base = select_sections(sin, sections)?;
    let (batch, q_heads, seq_len, head_dim) = q.shape().dims4()?;
    let cos_q = cos_base
        .clone()
        .unsqueeze(1)?
        .expand((batch, q_heads, seq_len, head_dim))?
        .contiguous()?;
    let sin_q = sin_base
        .clone()
        .unsqueeze(1)?
        .expand((batch, q_heads, seq_len, head_dim))?
        .contiguous()?;
    let q_rot = q.mul(&cos_q)?.add(&rotate_half(&q)?.mul(&sin_q)?)?;
    let (batch, k_heads, _, _) = k.shape().dims4()?;
    let cos_k = cos_base
        .unsqueeze(1)?
        .expand((batch, k_heads, seq_len, head_dim))?
        .contiguous()?;
    let sin_k = sin_base
        .unsqueeze(1)?
        .expand((batch, k_heads, seq_len, head_dim))?
        .contiguous()?;
    let k_rot = k.mul(&cos_k)?.add(&rotate_half(&k)?.mul(&sin_k)?)?;
    Ok((q_rot, k_rot))
}

fn select_sections(tensor: &Tensor, sections: &[usize]) -> Result<Tensor> {
    let (axes, _batch, _seq_len, dim) = tensor.shape().dims4()?;
    ensure!(axes == 3, "rotary tensor must have axis dimension 3");
    let mut offset = 0usize;
    let mut segments = Vec::with_capacity(sections.len());
    for (idx, width) in sections.iter().enumerate() {
        ensure!(*width > 0, "mrope section width must be > 0");
        let chunk = tensor.narrow(D::Minus1, offset, *width)?;
        let axis_slice = chunk.i(idx % axes)?.contiguous()?;
        segments.push(axis_slice);
        offset += *width;
    }
    ensure!(
        offset == dim,
        "mrope sections cover {offset} dims but tensor has {dim}"
    );
    let refs: Vec<&Tensor> = segments.iter().collect();
    Ok(Tensor::cat(&refs, D::Minus1)?)
}

fn validate_cache_shapes(
    cfg: &PaddleOcrVlConfig,
    key: &Tensor,
    value: &Tensor,
    batch: usize,
    head_dim: usize,
) -> Result<()> {
    let (cache_batch, cache_heads, cache_dim, _) = key.shape().dims4()?;
    ensure!(
        cache_batch == batch,
        "cache batch {} does not match input batch {}",
        cache_batch,
        batch
    );
    ensure!(
        cache_heads == cfg.num_attention_heads,
        "cache heads {} does not match {}",
        cache_heads,
        cfg.num_attention_heads
    );
    ensure!(
        cache_dim == head_dim,
        "cache head dim {} mismatch {}",
        cache_dim,
        head_dim
    );
    let value_dims = value.shape().dims();
    ensure!(value_dims.len() == 4, "cache value must be rank 4");
    ensure!(value_dims[0] == batch, "cache value batch mismatch");
    ensure!(
        value_dims[1] == cfg.num_attention_heads,
        "cache value heads {} mismatch {}",
        value_dims[1],
        cfg.num_attention_heads
    );
    ensure!(value_dims[3] == head_dim, "cache value dim mismatch");
    Ok(())
}

#[allow(unused_variables)]
fn flash_attention_forward(
    ctx: &AttentionContext<'_>,
    hidden_states: &Tensor,
    weights: &ErnieAttentionWeights,
    cos: &Tensor,
    sin: &Tensor,
    attn_bias: Option<&Tensor>,
    past_key_value: Option<&KvCacheEntry>,
    use_cache: bool,
) -> Result<Option<(Tensor, Option<KvCacheChunk>)>> {
    #[cfg(not(feature = "flash-attn"))]
    {
        let _ = (
            ctx,
            hidden_states,
            weights,
            cos,
            sin,
            attn_bias,
            past_key_value,
            use_cache,
        );
        return Ok(None);
    }
    #[cfg(feature = "flash-attn")]
    {
        if attn_bias.is_some() || past_key_value.is_some() {
            return Ok(None);
        }
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();
        if !supports_flash_attention(ctx.cfg, device, dtype) {
            return Ok(None);
        }
        let cfg = ctx.cfg;
        let (batch, seq_len, hidden_size) = hidden_states.shape().dims3()?;
        ensure!(
            hidden_size == cfg.hidden_size,
            "hidden size mismatch: cfg {} tensor {}",
            cfg.hidden_size,
            hidden_size
        );
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.resolved_num_key_value_heads();

        let q = apply_linear(hidden_states, &weights.q_proj)?
            .reshape((batch, seq_len, num_heads, head_dim))?
            .to_dtype(dtype)?
            .to_device(device)?;
        let k = apply_linear(hidden_states, &weights.k_proj)?
            .reshape((batch, seq_len, num_kv_heads, head_dim))?
            .to_dtype(dtype)?
            .to_device(device)?;
        let v = apply_linear(hidden_states, &weights.v_proj)?
            .reshape((batch, seq_len, num_kv_heads, head_dim))?
            .to_dtype(dtype)?
            .to_device(device)?;

        let sections = ctx.rotary.doubled_sections();
        let (mut q, mut k) = apply_multimodal_rotary(q, k, cos, sin, sections)?;

        let repeats = num_heads / num_kv_heads;
        k = repeat_kv(&k, repeats)?;
        let mut v = repeat_kv(&v, repeats)?;

        q = q.contiguous()?;
        k = k.contiguous()?;
        v = v.contiguous()?;

        let present = if use_cache {
            let key_t = k.permute((0, 1, 3, 2))?.contiguous()?;
            Some(KvCacheChunk::new(key_t, v.clone())?)
        } else {
            None
        };

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v_t = v.transpose(1, 2)?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let causal = true;
        let attn = flash_attn(&q, &k, &v_t, scale, causal)?;
        let attn = attn
            .transpose(1, 2)?
            .reshape((batch, seq_len, num_heads * head_dim))?;
        let output = apply_linear(&attn, &weights.o_proj)?;
        Ok(Some((output, present)))
    }
}
