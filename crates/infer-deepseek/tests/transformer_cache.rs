use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use deepseek_ocr_infer_deepseek::transformer::cache::{DynamicCache, KvCacheChunk, LayerKvCache};

fn make_chunk(
    device: &Device,
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
) -> Result<KvCacheChunk> {
    let key_t_shape = (batch, heads, dim, seq);
    let key_t = Tensor::zeros(key_t_shape, DType::F32, device)?;
    let value_shape = (batch, heads, seq, dim);
    let value = Tensor::zeros(value_shape, DType::F32, device)?;
    KvCacheChunk::new(key_t, value)
}

#[test]
fn layer_cache_auto_resizes_and_tracks_len() -> Result<()> {
    let device = Device::Cpu;
    let mut cache = LayerKvCache::new();
    let chunk = make_chunk(&device, 1, 2, 3, 4)?;
    cache.append_chunk(1, chunk)?;
    assert_eq!(cache.len(), 2);
    assert!(cache.get(0).is_none());
    let entry = cache.get(1).expect("cache entry");
    assert_eq!(entry.seq_len(), 3);
    assert_eq!(cache.seq_len(), Some(3));
    Ok(())
}

#[test]
fn layer_cache_rejects_incompatible_dimensions() -> Result<()> {
    let device = Device::Cpu;
    let mut cache = LayerKvCache::with_num_layers(1);
    let chunk = make_chunk(&device, 1, 2, 3, 4)?;
    cache.append_chunk(0, chunk)?;

    let mismatched = make_chunk(&device, 1, 3, 1, 4)?;
    let err = cache
        .append_chunk(0, mismatched)
        .expect_err("expected shape mismatch");
    assert!(err.to_string().contains("chunk heads"));
    Ok(())
}

#[test]
fn dynamic_cache_tracks_sequence_growth() -> Result<()> {
    let device = Device::Cpu;
    let mut cache = DynamicCache::with_num_layers(3);
    let chunk_prefill = make_chunk(&device, 1, 2, 3, 4)?;
    cache.append(0, chunk_prefill.clone())?;
    cache.append(1, chunk_prefill.clone())?;
    assert_eq!(cache.seq_len(), Some(3));

    let chunk_step = make_chunk(&device, 1, 2, 1, 4)?;
    cache.append(0, chunk_step.clone())?;
    assert_eq!(cache.seq_len(), Some(4));

    let short_prefill = make_chunk(&device, 1, 2, 2, 4)?;
    let err = cache
        .append(2, short_prefill)
        .expect_err("expected seq_len decrease to fail");
    assert!(err.to_string().contains("seq_len decreased"));

    cache.append(1, chunk_step)?;
    assert_eq!(cache.seq_len(), Some(4));
    Ok(())
}

#[test]
fn prompt_guard_clears_on_drop() -> Result<()> {
    let device = Device::Cpu;
    let mut cache = DynamicCache::with_num_layers(1);
    {
        let mut guard = cache.prompt_guard();
        let chunk = make_chunk(&device, 1, 2, 3, 4)?;
        guard.cache().append(0, chunk)?;
        assert_eq!(guard.cache().seq_len(), Some(3));
    }
    assert!(cache.seq_len().is_none());
    assert!(cache.layers().iter().all(|entry| entry.is_none()));
    Ok(())
}

#[test]
fn prompt_guard_runs_reset_hook() -> Result<()> {
    use std::cell::Cell;

    let device = Device::Cpu;
    let mut cache = DynamicCache::with_num_layers(1);
    let flag = Cell::new(false);
    {
        let mut guard = cache.prompt_guard_with_reset(|| flag.set(true));
        let chunk = make_chunk(&device, 1, 2, 3, 4)?;
        guard.cache().append(0, chunk)?;
    }
    assert!(flag.get());
    Ok(())
}
