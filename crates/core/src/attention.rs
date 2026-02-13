use anyhow::{Result, ensure};
use candle_core::{Tensor, shape::D};

use crate::cache::KvCacheEntry;

#[derive(Debug, Clone)]
pub struct CacheChunkViews {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
    pub total_len: usize,
}

pub fn collect_cache_chunk_views(
    entry: &KvCacheEntry,
    batch: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<CacheChunkViews> {
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let mut total_len = 0usize;

    for chunk in entry.chunks() {
        let key = if chunk.key_t.is_contiguous() {
            chunk.key_t.clone()
        } else {
            chunk.key_t.contiguous()?
        };
        let value = if chunk.value.is_contiguous() {
            chunk.value.clone()
        } else {
            chunk.value.contiguous()?
        };

        let (cache_batch, cache_heads, cache_key_dim, chunk_len) = key.shape().dims4()?;
        ensure!(
            cache_batch == batch,
            "cache key batch {} does not match expected {}",
            cache_batch,
            batch
        );
        ensure!(
            cache_heads == heads,
            "cache key heads {} does not match expected {}",
            cache_heads,
            heads
        );
        ensure!(
            cache_key_dim == key_dim,
            "cache key dim {} does not match expected {}",
            cache_key_dim,
            key_dim
        );

        let (value_batch, value_heads, value_len, cache_value_dim) = value.shape().dims4()?;
        ensure!(
            value_batch == batch,
            "cache value batch {} does not match expected {}",
            value_batch,
            batch
        );
        ensure!(
            value_heads == heads,
            "cache value heads {} does not match expected {}",
            value_heads,
            heads
        );
        ensure!(
            value_len == chunk_len,
            "cache value len {} does not match key len {}",
            value_len,
            chunk_len
        );
        ensure!(
            cache_value_dim == value_dim,
            "cache value dim {} does not match expected {}",
            cache_value_dim,
            value_dim
        );

        total_len += chunk_len;
        keys.push(key);
        values.push(value);
    }

    ensure!(
        total_len == entry.seq_len(),
        "cache chunks cover {} tokens but cache entry records {}",
        total_len,
        entry.seq_len()
    );

    Ok(CacheChunkViews {
        keys,
        values,
        total_len,
    })
}

pub fn scores_from_cache_chunks(
    query: &Tensor,
    cache_keys: &[Tensor],
    new_key: &Tensor,
) -> Result<Tensor> {
    if cache_keys.is_empty() {
        return Ok(query.matmul(new_key)?);
    }

    let mut score_parts = Vec::with_capacity(cache_keys.len() + 1);
    for cache_key in cache_keys {
        score_parts.push(query.matmul(cache_key)?);
    }
    score_parts.push(query.matmul(new_key)?);

    if score_parts.len() == 1 {
        Ok(score_parts.pop().expect("single score chunk"))
    } else {
        let refs: Vec<&Tensor> = score_parts.iter().collect();
        Ok(Tensor::cat(&refs, D::Minus1)?)
    }
}

pub fn output_from_cache_chunks(
    attn_weights: &Tensor,
    cache_values: &[Tensor],
    new_values: &Tensor,
    past_len: usize,
    new_len: usize,
) -> Result<Tensor> {
    if cache_values.is_empty() || past_len == 0 {
        return Ok(attn_weights.matmul(new_values)?);
    }

    let mut offset = 0usize;
    let mut cached: Option<Tensor> = None;
    for cache_value in cache_values {
        let chunk_len = cache_value.dim(D::Minus2)?;
        let chunk_weights = attn_weights.narrow(D::Minus1, offset, chunk_len)?;
        offset += chunk_len;
        let contribution = chunk_weights.matmul(cache_value)?;
        cached = Some(match cached {
            Some(existing) => existing.add(&contribution)?,
            None => contribution,
        });
    }

    ensure!(
        offset == past_len,
        "cache coverage {} does not match past_len {}",
        offset,
        past_len
    );
    let current = attn_weights
        .narrow(D::Minus1, past_len, new_len)?
        .matmul(new_values)?;

    if let Some(cached) = cached {
        Ok(cached.add(&current)?)
    } else {
        Ok(current)
    }
}

