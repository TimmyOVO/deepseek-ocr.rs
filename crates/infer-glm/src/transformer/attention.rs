use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor, shape::D};
use candle_nn::ops::softmax;
use deepseek_ocr_core::cache::{KvCacheChunk, KvCacheEntry};

use super::{
    ops::{compute_dtype_for, maybe_cast, repeat_kv, rotate_half_last_dim},
    weights::GlmTextAttentionWeights,
};

pub struct TextAttentionContext {
    pub num_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub scaling: f64,
}

pub fn attention_forward(
    ctx: &TextAttentionContext,
    hidden_states: &Tensor,
    weights: &GlmTextAttentionWeights,
    cos: &Tensor,
    sin: &Tensor,
    attn_bias: Option<&Tensor>,
    past_key_value: Option<&KvCacheEntry>,
    use_cache: bool,
) -> Result<(Tensor, Option<KvCacheChunk>)> {
    let (batch, seq_len, _hidden_size) = hidden_states.shape().dims3()?;

    let q = weights
        .q_proj
        .forward_3d(hidden_states)?
        .reshape((batch, seq_len, ctx.num_heads, ctx.head_dim))?
        .permute((0, 2, 1, 3))?;
    let k = weights
        .k_proj
        .forward_3d(hidden_states)?
        .reshape((batch, seq_len, ctx.num_key_value_heads, ctx.head_dim))?
        .permute((0, 2, 1, 3))?;
    let v = weights
        .v_proj
        .forward_3d(hidden_states)?
        .reshape((batch, seq_len, ctx.num_key_value_heads, ctx.head_dim))?
        .permute((0, 2, 1, 3))?;

    let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin)?;

    let repeats = ctx.num_heads / ctx.num_key_value_heads;
    let mut k = repeat_kv(&k, repeats)?;
    let mut v = repeat_kv(&v, repeats)?;

    let mut cache_key_t_view: Option<Tensor> = None;
    let mut cache_value_view: Option<Tensor> = None;
    let past_len = if let Some(entry) = past_key_value {
        let key_view = entry.key_view()?;
        let value_view = entry.value_view()?;
        validate_cache_shapes(ctx, &key_view, &value_view, batch)?;
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
    let score_dtype = compute_dtype_for(&q);
    let q_for_scores = maybe_cast(&q, score_dtype)?;
    let k_t_for_scores = maybe_cast(&k_t, score_dtype)?;
    let attn_scores = if let Some(cache_key) = cache_key_t_view.as_ref() {
        if past_len > 0 {
            let cache_key = maybe_cast(&cache_key.contiguous()?, score_dtype)?;
            let full_k_t = Tensor::cat(&[&cache_key, &k_t_for_scores], D::Minus1)?;
            q_for_scores.matmul(&full_k_t)?
        } else {
            q_for_scores.matmul(&k_t_for_scores)?
        }
    } else {
        q_for_scores.matmul(&k_t_for_scores)?
    };

    let mut attn_scores = (attn_scores / ctx.scaling.sqrt())?;
    if let Some(bias) = attn_bias {
        let bias = maybe_cast(bias, attn_scores.dtype())?;
        attn_scores = attn_scores.broadcast_add(&bias)?;
    }

    let attn_weights = softmax(&attn_scores, D::Minus1).context("attention softmax failed")?;
    let value_dtype = attn_weights.dtype();
    let v_for_values = maybe_cast(&v, value_dtype)?;
    let mut attn_output = if let Some(cache_value) = cache_value_view.as_ref() {
        if past_len > 0 {
            let cache_value = maybe_cast(&cache_value.contiguous()?, value_dtype)?;
            let full_v = Tensor::cat(&[&cache_value, &v_for_values], D::Minus2)?;
            attn_weights.matmul(&full_v)?
        } else {
            attn_weights.matmul(&v_for_values)?
        }
    } else {
        attn_weights.matmul(&v_for_values)?
    };
    if attn_output.dtype() != q.dtype() {
        attn_output = attn_output.to_dtype(q.dtype())?;
    }

    let present = if use_cache {
        Some(KvCacheChunk::new(k_t.clone(), v.clone())?)
    } else {
        None
    };

    let context = attn_output
        .permute((0, 2, 1, 3))?
        .reshape((batch, seq_len, ctx.num_heads * ctx.head_dim))?;
    let output = weights.o_proj.forward_3d(&context)?;
    Ok((output, present))
}

pub fn build_attention_bias(
    attention_mask: Option<&Tensor>,
    batch: usize,
    q_len: usize,
    k_len: usize,
    past_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut bias = if q_len == 0 || k_len == 0 {
        Tensor::zeros((batch, 1, q_len, k_len), dtype, device)?
    } else if past_len == 0 && q_len == k_len {
        let mut data = vec![0f32; batch * q_len * k_len];
        for b in 0..batch {
            for q in 0..q_len {
                for k in (q + 1)..k_len {
                    data[b * q_len * k_len + q * k_len + k] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_vec(data, (batch, 1, q_len, k_len), device)?.to_dtype(dtype)?
    } else {
        Tensor::zeros((batch, 1, q_len, k_len), dtype, device)?
    };

    if let Some(mask) = attention_mask {
        let (mask_batch, mask_len) = mask.shape().dims2()?;
        ensure!(
            mask_batch == batch,
            "attention mask batch {} != {}",
            mask_batch,
            batch
        );
        ensure!(
            mask_len == k_len,
            "attention mask len {} != {}",
            mask_len,
            k_len
        );
        let mask = if mask.dtype() == DType::U8 {
            mask.clone()
        } else {
            mask.to_dtype(DType::U8)?
        };
        let mask = mask.to_vec2::<u8>()?;
        let mut pad = vec![0f32; batch * q_len * k_len];
        for b in 0..batch {
            for q in 0..q_len {
                for k in 0..k_len {
                    if mask[b][k] == 0 {
                        pad[b * q_len * k_len + q * k_len + k] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let pad = Tensor::from_vec(pad, (batch, 1, q_len, k_len), device)?.to_dtype(dtype)?;
        bias = bias.broadcast_add(&pad)?;
    }

    Ok(bias)
}

fn apply_rotary_pos_emb(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let q_heads = q.dim(1)?;
    let k_heads = k.dim(1)?;
    let cos_q = expand_interleaved(cos, q_heads)?;
    let sin_q = expand_interleaved(sin, q_heads)?;
    let rotary_dim = cos_q.dim(D::Minus1)?;

    let q_rot = q.narrow(D::Minus1, 0, rotary_dim)?;
    let q_pass = q.narrow(D::Minus1, rotary_dim, q.dim(D::Minus1)? - rotary_dim)?;
    let k_rot = k.narrow(D::Minus1, 0, rotary_dim)?;
    let k_pass = k.narrow(D::Minus1, rotary_dim, k.dim(D::Minus1)? - rotary_dim)?;

    let q_embed = q_rot
        .broadcast_mul(&cos_q)?
        .add(&rotate_half_last_dim(&q_rot)?.broadcast_mul(&sin_q)?)?;

    let (cos_k, sin_k) = if k_heads == q_heads {
        (cos_q, sin_q)
    } else {
        (expand_interleaved(cos, k_heads)?, expand_interleaved(sin, k_heads)?)
    };
    let k_embed = k_rot
        .broadcast_mul(&cos_k)?
        .add(&rotate_half_last_dim(&k_rot)?.broadcast_mul(&sin_k)?)?;

    Ok((
        Tensor::cat(&[q_embed, q_pass], D::Minus1)?,
        Tensor::cat(&[k_embed, k_pass], D::Minus1)?,
    ))
}

fn expand_interleaved(base: &Tensor, heads: usize) -> Result<Tensor> {
    let (batch, seq, head_dim) = base.shape().dims3()?;
    ensure!(head_dim.is_multiple_of(2), "rope head dim must be even");
    let half = head_dim / 2;

    let interleaved = base
        .narrow(D::Minus1, 0, half)?
        .unsqueeze(D::Minus1)?
        .expand((batch, seq, half, 2))?
        .reshape((batch, seq, head_dim))?;

    Ok(interleaved.unsqueeze(1)?.expand((batch, heads, seq, head_dim))?)
}

fn validate_cache_shapes(
    ctx: &TextAttentionContext,
    key: &Tensor,
    value: &Tensor,
    batch: usize,
) -> Result<()> {
    let (cache_batch, cache_heads, cache_dim, _) = key.shape().dims4()?;
    ensure!(cache_batch == batch, "cache batch mismatch");
    ensure!(cache_heads == ctx.num_heads, "cache heads mismatch");
    ensure!(cache_dim == ctx.head_dim, "cache head dim mismatch");

    let value_dims = value.shape().dims();
    ensure!(value_dims.len() == 4, "cache value must be rank 4");
    ensure!(value_dims[0] == batch, "cache value batch mismatch");
    ensure!(value_dims[1] == ctx.num_heads, "cache value heads mismatch");
    ensure!(value_dims[3] == ctx.head_dim, "cache value head dim mismatch");
    Ok(())
}
