use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor, shape::D};
use candle_nn::ops::{softmax, softmax_last_dim};
use deepseek_ocr_core::benchmark::Timer;
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

pub struct AttentionForwardArgs<'a> {
    pub hidden_states: &'a Tensor,
    pub weights: &'a GlmTextAttentionWeights,
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
    pub attn_bias: Option<&'a Tensor>,
    pub past_key_value: Option<&'a KvCacheEntry>,
    pub use_cache: bool,
}

pub fn attention_forward(
    ctx: &TextAttentionContext,
    args: AttentionForwardArgs<'_>,
) -> Result<(Tensor, Option<KvCacheChunk>)> {
    let AttentionForwardArgs {
        hidden_states,
        weights,
        cos,
        sin,
        attn_bias,
        past_key_value,
        use_cache,
    } = args;

    let (batch, seq_len, _hidden_size) = hidden_states.shape().dims3()?;

    let qkv_timer = Timer::new("text.attn.qkv_proj");
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
    qkv_timer.finish(|_| {});

    let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin)?;

    let repeats = ctx.num_heads / ctx.num_key_value_heads;
    let mut k = repeat_kv(&k, repeats)?;
    let mut v = repeat_kv(&v, repeats)?;

    let mut cache_keys: Option<Vec<Tensor>> = None;
    let mut cache_values: Option<Vec<Tensor>> = None;
    let past_len = if let Some(entry) = past_key_value {
        let key_chunks: Vec<Tensor> = entry
            .chunks()
            .iter()
            .map(|chunk| chunk.key_t.clone())
            .collect();
        let value_chunks: Vec<Tensor> = entry
            .chunks()
            .iter()
            .map(|chunk| chunk.value.clone())
            .collect();
        validate_cache_chunk_shapes(ctx, &key_chunks, &value_chunks, batch)?;
        cache_keys = Some(key_chunks);
        cache_values = Some(value_chunks);
        entry.seq_len()
    } else {
        0
    };

    let q = q.contiguous()?;
    k = k.contiguous()?;
    v = v.contiguous()?;

    let k_t = k.permute((0, 1, 3, 2))?.contiguous()?;
    let score_dtype = compute_dtype_for(&q);
    let cache_dtype = score_dtype;
    let q_for_scores = maybe_cast(&q, score_dtype)?;
    let q_for_scores = (q_for_scores * (1.0f64 / ctx.scaling.sqrt()))?;
    let k_t_for_scores = maybe_cast(&k_t, score_dtype)?;
    let attn_scores = if let Some(cache_key_chunks) = cache_keys.as_ref() {
        if past_len > 0 {
            let mut cache_scores_parts = Vec::with_capacity(cache_key_chunks.len());
            for cache_key_chunk in cache_key_chunks {
                let cache_key_contig_timer = Timer::new("text.attn.cache_key.contiguous");
                let cache_key = cache_key_chunk.contiguous()?;
                cache_key_contig_timer.finish(|_| {});
                let cache_key = maybe_cast(&cache_key, score_dtype)?;

                let score_cache_mm_timer = Timer::new("text.attn.score.matmul_cache");
                let cache_scores = q_for_scores.matmul(&cache_key)?;
                score_cache_mm_timer.finish(|_| {});
                cache_scores_parts.push(cache_scores);
            }
            let cache_scores = if cache_scores_parts.len() == 1 {
                cache_scores_parts.pop().expect("one cache score chunk")
            } else {
                let refs: Vec<&Tensor> = cache_scores_parts.iter().collect();
                Tensor::cat(&refs, D::Minus1)?
            };

            let score_new_mm_timer = Timer::new("text.attn.score.matmul_new");
            let new_scores = q_for_scores.matmul(&k_t_for_scores)?;
            score_new_mm_timer.finish(|_| {});
            Tensor::cat(&[&cache_scores, &new_scores], D::Minus1)?
        } else {
            q_for_scores.matmul(&k_t_for_scores)?
        }
    } else {
        q_for_scores.matmul(&k_t_for_scores)?
    };

    let mut attn_scores = attn_scores;
    if let Some(bias) = attn_bias {
        let bias = maybe_cast(bias, attn_scores.dtype())?;
        attn_scores = attn_scores.broadcast_add(&bias)?;
    }

    let softmax_timer = Timer::new("text.attn.softmax");
    let attn_weights = if seq_len == 1 {
        softmax(&attn_scores, D::Minus1)
    } else {
        softmax_last_dim(&attn_scores)
    }
    .context("attention softmax failed")?;
    softmax_timer.finish(|_| {});
    let value_dtype = attn_weights.dtype();
    let v_for_values = maybe_cast(&v, value_dtype)?;
    let mut attn_output = if let Some(cache_value_chunks) = cache_values.as_ref() {
        if past_len > 0 {
            let cache_lens: Vec<usize> = cache_value_chunks
                .iter()
                .map(|chunk| chunk.dim(D::Minus2))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let cache_len = cache_lens.iter().sum::<usize>();
            let new_len = v_for_values.dim(D::Minus2)?;
            let cache_weights = attn_weights.narrow(D::Minus1, 0, cache_len)?;
            let new_weights = attn_weights.narrow(D::Minus1, cache_len, new_len)?;

            let mut cache_outputs = Vec::with_capacity(cache_value_chunks.len());
            let mut offset = 0usize;
            for (chunk_idx, cache_value_chunk) in cache_value_chunks.iter().enumerate() {
                let chunk_len = cache_lens[chunk_idx];
                let chunk_weights = cache_weights.narrow(D::Minus1, offset, chunk_len)?;
                offset += chunk_len;

                let cache_value_contig_timer = Timer::new("text.attn.cache_value.contiguous");
                let cache_value = cache_value_chunk.contiguous()?;
                cache_value_contig_timer.finish(|_| {});
                let cache_value = maybe_cast(&cache_value, value_dtype)?;

                let value_cache_mm_timer = Timer::new("text.attn.value.matmul_cache");
                let cache_output = chunk_weights.matmul(&cache_value)?;
                value_cache_mm_timer.finish(|_| {});
                cache_outputs.push(cache_output);
            }
            let mut cache_output = cache_outputs
                .pop()
                .expect("at least one cache output chunk");
            for part in cache_outputs {
                cache_output = part.add(&cache_output)?;
            }

            let value_new_mm_timer = Timer::new("text.attn.value.matmul_new");
            let new_output = new_weights.matmul(&v_for_values)?;
            value_new_mm_timer.finish(|_| {});
            cache_output.add(&new_output)?
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
        let cache_k_t = if k_t.dtype() == cache_dtype {
            k_t.clone()
        } else {
            k_t.to_dtype(cache_dtype)?
        };
        let cache_v = if v.dtype() == cache_dtype {
            v.clone()
        } else {
            v.to_dtype(cache_dtype)?
        };
        Some(KvCacheChunk::new(cache_k_t, cache_v)?)
    } else {
        None
    };

    let context = attn_output.permute((0, 2, 1, 3))?.reshape((
        batch,
        seq_len,
        ctx.num_heads * ctx.head_dim,
    ))?;
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

fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
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
        (
            expand_interleaved(cos, k_heads)?,
            expand_interleaved(sin, k_heads)?,
        )
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

    Ok(interleaved
        .unsqueeze(1)?
        .expand((batch, heads, seq, head_dim))?)
}

fn validate_cache_chunk_shapes(
    ctx: &TextAttentionContext,
    key_chunks: &[Tensor],
    value_chunks: &[Tensor],
    batch: usize,
) -> Result<()> {
    ensure!(
        !key_chunks.is_empty() && key_chunks.len() == value_chunks.len(),
        "cache chunk count mismatch"
    );
    for (key, value) in key_chunks.iter().zip(value_chunks.iter()) {
        let (cache_batch, cache_heads, cache_dim, cache_seq) = key.shape().dims4()?;
        ensure!(cache_batch == batch, "cache batch mismatch");
        ensure!(cache_heads == ctx.num_heads, "cache heads mismatch");
        ensure!(cache_dim == ctx.head_dim, "cache head dim mismatch");

        let (value_batch, value_heads, value_seq, value_dim) = value.shape().dims4()?;
        ensure!(value_batch == batch, "cache value batch mismatch");
        ensure!(value_heads == ctx.num_heads, "cache value heads mismatch");
        ensure!(value_dim == ctx.head_dim, "cache value head dim mismatch");
        ensure!(value_seq == cache_seq, "cache key/value seq mismatch");
    }
    Ok(())
}
