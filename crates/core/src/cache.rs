use anyhow::{Context, Result, ensure};
use candle_core::{Tensor, shape::D};
use std::boxed::Box;

use crate::benchmark::Timer;

#[cfg(feature = "memlog")]
use crate::memlog;

/// Newly computed K/V tensors to append to the cache.
///
/// Keys are stored transposed as `[batch, heads, dim, seq]` so we can reuse them directly in
/// attention matmuls without an extra transpose per decode step.
#[derive(Debug, Clone)]
pub struct KvCacheChunk {
    pub key_t: Tensor,
    pub value: Tensor,
}

impl KvCacheChunk {
    pub fn new(key_t: Tensor, value: Tensor) -> Result<Self> {
        ensure!(
            key_t.rank() == 4,
            "expected key chunk tensor with rank 4 [batch, heads, dim, seq], got rank {}",
            key_t.rank()
        );
        ensure!(
            value.rank() == 4,
            "expected value chunk tensor with rank 4 [batch, heads, seq, dim], got rank {}",
            value.rank()
        );
        let (key_batch, key_heads, _key_dim, key_seq) =
            key_t.shape().dims4().context("key chunk must be 4D")?;
        let (val_batch, val_heads, val_seq, _) =
            value.shape().dims4().context("value chunk must be 4D")?;
        ensure!(
            key_batch == val_batch,
            "chunk batch mismatch between key ({key_batch}) and value ({val_batch})"
        );
        ensure!(
            key_heads == val_heads,
            "chunk heads mismatch between key ({key_heads}) and value ({val_heads})"
        );
        ensure!(
            key_seq == val_seq,
            "chunk sequence mismatch between key ({key_seq}) and value ({val_seq})"
        );
        Ok(Self { key_t, value })
    }

    pub fn seq_len(&self) -> usize {
        self.key_t
            .dim(candle_core::shape::D::Minus1)
            .expect("chunk tensors are validated to rank 4")
    }
}

/// Growable key/value cache for a single transformer layer.
#[derive(Debug, Clone)]
pub struct KvCacheEntry {
    chunks: Vec<KvCacheChunk>,
    len: usize,
}

impl KvCacheEntry {
    pub fn from_chunk(chunk: KvCacheChunk) -> Result<Self> {
        let len = chunk.seq_len();
        Ok(Self {
            chunks: vec![chunk],
            len,
        })
    }

    pub fn chunks(&self) -> &[KvCacheChunk] {
        &self.chunks
    }

    fn first_chunk(&self) -> Option<&KvCacheChunk> {
        self.chunks.first()
    }

    fn validate_chunk(&self, chunk: &KvCacheChunk) -> Result<()> {
        let Some(first) = self.first_chunk() else {
            return Ok(());
        };
        let (batch, heads, key_dim, _) = first
            .key_t
            .shape()
            .dims4()
            .context("cache key tensor must be 4D")?;
        let (chunk_batch, chunk_heads, chunk_key_dim, chunk_len) = chunk
            .key_t
            .shape()
            .dims4()
            .context("key chunk must be 4D")?;
        ensure!(
            chunk_batch == batch,
            "chunk batch {} does not match cache batch {}",
            chunk_batch,
            batch
        );
        ensure!(
            chunk_heads == heads,
            "chunk heads {} does not match cache heads {}",
            chunk_heads,
            heads
        );
        ensure!(
            chunk_key_dim == key_dim,
            "chunk key dim {} does not match cache key dim {}",
            chunk_key_dim,
            key_dim
        );
        ensure!(
            chunk.key_t.dtype() == first.key_t.dtype(),
            "chunk dtype {:?} does not match cache dtype {:?}",
            chunk.key_t.dtype(),
            first.key_t.dtype()
        );
        ensure!(
            chunk.key_t.device().location() == first.key_t.device().location(),
            "chunk device {:?} does not match cache device {:?}",
            chunk.key_t.device(),
            first.key_t.device()
        );
        let (_, _, _value_seq, value_dim) = self
            .first_chunk()
            .expect("first chunk must exist")
            .value
            .shape()
            .dims4()
            .context("value tensor must be 4D")?;
        let (chunk_val_batch, chunk_val_heads, chunk_val_seq, chunk_val_dim) = chunk
            .value
            .shape()
            .dims4()
            .context("chunk value must be 4D")?;
        ensure!(
            chunk_val_batch == batch,
            "chunk value batch {} does not match cache batch {}",
            chunk_val_batch,
            batch
        );
        ensure!(
            chunk_val_heads == heads,
            "chunk value heads {} does not match cache heads {}",
            chunk_val_heads,
            heads
        );
        ensure!(
            chunk_val_seq == chunk_len,
            "chunk value seq {} does not match key seq {}",
            chunk_val_seq,
            chunk_len
        );
        ensure!(
            chunk_val_dim == value_dim,
            "chunk value dim {} does not match cache value dim {}",
            chunk_val_dim,
            value_dim
        );
        ensure!(
            chunk.value.dtype() == first.value.dtype(),
            "chunk value dtype {:?} does not match cache value dtype {:?}",
            chunk.value.dtype(),
            first.value.dtype()
        );
        ensure!(
            chunk.value.device().location() == first.value.device().location(),
            "chunk value device {:?} does not match cache value device {:?}",
            chunk.value.device(),
            first.value.device()
        );
        Ok(())
    }

    pub fn append(&mut self, chunk: &KvCacheChunk) -> Result<()> {
        let append_total_timer = Timer::new("cache.entry.append.total");
        let chunk = if let Some(first) = self.first_chunk() {
            if chunk.key_t.dtype() != first.key_t.dtype() || chunk.value.dtype() != first.value.dtype() {
                KvCacheChunk::new(
                    chunk.key_t.to_dtype(first.key_t.dtype())?,
                    chunk.value.to_dtype(first.value.dtype())?,
                )?
            } else {
                chunk.clone()
            }
        } else {
            chunk.clone()
        };
        self.validate_chunk(&chunk)?;
        let chunk_len = chunk.seq_len();
        if chunk_len == 0 {
            return Ok(());
        }
        self.chunks.push(chunk);
        self.len += chunk_len;
        append_total_timer.finish(|_| {});
        Ok(())
    }

    pub fn key_view(&self) -> Result<Tensor> {
        if self.chunks.is_empty() {
            anyhow::bail!("cache entry has no key chunks")
        }
        if self.chunks.len() == 1 {
            return Ok(self.chunks[0].key_t.clone());
        }
        let refs: Vec<&Tensor> = self.chunks.iter().map(|chunk| &chunk.key_t).collect();
        Ok(Tensor::cat(&refs, D::Minus1)?)
    }

    pub fn value_view(&self) -> Result<Tensor> {
        if self.chunks.is_empty() {
            anyhow::bail!("cache entry has no value chunks")
        }
        if self.chunks.len() == 1 {
            return Ok(self.chunks[0].value.clone());
        }
        let refs: Vec<&Tensor> = self.chunks.iter().map(|chunk| &chunk.value).collect();
        Ok(Tensor::cat(&refs, D::Minus2)?)
    }

    pub fn seq_len(&self) -> usize {
        self.len
    }

    #[cfg(feature = "memlog")]
    pub fn storage_bytes(&self) -> usize {
        self.chunks
            .iter()
            .map(|chunk| memlog::tensor_bytes(&chunk.key_t) + memlog::tensor_bytes(&chunk.value))
            .sum()
    }
}

/// Collection of per-layer KV cache entries.
#[derive(Debug, Clone, Default)]
pub struct LayerKvCache {
    entries: Vec<Option<KvCacheEntry>>,
}

impl LayerKvCache {
    /// Create an empty cache with no preallocated layers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a cache with the given number of layers preallocated.
    pub fn with_num_layers(num_layers: usize) -> Self {
        Self {
            entries: vec![None; num_layers],
        }
    }

    /// Current number of layer slots tracked by this cache (including empty ones).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no layer has cached values yet.
    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(|entry| entry.is_none())
    }

    /// Return the cached entry for `layer_idx`, if any.
    pub fn get(&self, layer_idx: usize) -> Option<&KvCacheEntry> {
        self.entries.get(layer_idx).and_then(|entry| entry.as_ref())
    }

    /// Append the provided chunk to the cache entry for `layer_idx`, creating it if needed.
    pub fn append_chunk(&mut self, layer_idx: usize, chunk: KvCacheChunk) -> Result<()> {
        if layer_idx >= self.entries.len() {
            self.entries.resize_with(layer_idx + 1, || None);
        }
        if let Some(existing) = self.entries[layer_idx].as_mut() {
            existing.append(&chunk)
        } else {
            let entry = KvCacheEntry::from_chunk(chunk)?;
            #[cfg(feature = "memlog")]
            {
                memlog::add_kv(entry.storage_bytes());
            }
            self.entries[layer_idx] = Some(entry);
            Ok(())
        }
    }

    /// Clears all cached layers but retains the allocated capacity.
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            #[cfg(feature = "memlog")]
            if let Some(existing) = entry.as_ref() {
                memlog::sub_kv(existing.storage_bytes());
            }
            *entry = None;
        }
    }

    /// Ensure the cache tracks at least `total_layers` layers.
    pub fn ensure_layers(&mut self, total_layers: usize) {
        if self.entries.len() < total_layers {
            self.entries.resize_with(total_layers, || None);
        }
    }

    /// Iterate over layer entries.
    pub fn iter(&self) -> impl Iterator<Item = Option<&KvCacheEntry>> {
        self.entries.iter().map(|entry| entry.as_ref())
    }

    /// Borrow the raw entries vector.
    pub fn entries(&self) -> &[Option<KvCacheEntry>] {
        &self.entries
    }

    /// Mutable access to the raw entries vector.
    pub fn entries_mut(&mut self) -> &mut [Option<KvCacheEntry>] {
        &mut self.entries
    }

    /// Consume the cache and return the underlying entries vector.
    pub fn into_entries(self) -> Vec<Option<KvCacheEntry>> {
        self.entries
    }

    /// Best-effort sequence length derived from the first populated layer.
    pub fn seq_len(&self) -> Option<usize> {
        self.entries
            .iter()
            .filter_map(|entry| entry.as_ref().map(|kv| kv.seq_len()))
            .max()
    }
}

/// Dynamic cache that can grow across decoding steps.
#[derive(Debug, Clone, Default)]
pub struct DynamicCache {
    layers: LayerKvCache,
    seq_len: Option<usize>,
}

/// Clears a [`DynamicCache`] when dropped, ensuring prompt-scoped state cannot leak. Optionally
/// runs a caller-provided reset hook (e.g., to drop RoPE tables) after the cache has been cleared.
pub struct PromptCacheGuard<'a> {
    cache: &'a mut DynamicCache,
    rope_reset: Option<Box<dyn FnOnce() + 'a>>,
}

impl<'a> PromptCacheGuard<'a> {
    pub fn new(cache: &'a mut DynamicCache) -> Self {
        Self {
            cache,
            rope_reset: None,
        }
    }

    pub fn with_rope_reset<F>(cache: &'a mut DynamicCache, reset: F) -> Self
    where
        F: FnOnce() + 'a,
    {
        Self {
            cache,
            rope_reset: Some(Box::new(reset)),
        }
    }

    pub fn cache(&mut self) -> &mut DynamicCache {
        self.cache
    }
}

impl Drop for PromptCacheGuard<'_> {
    fn drop(&mut self) {
        self.cache.clear();
        if let Some(reset) = self.rope_reset.take() {
            reset();
        }
    }
}

impl DynamicCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_num_layers(num_layers: usize) -> Self {
        Self {
            layers: LayerKvCache::with_num_layers(num_layers),
            seq_len: None,
        }
    }

    /// Returns the cached entry for `layer_idx`, if present.
    pub fn get(&self, layer_idx: usize) -> Option<&KvCacheEntry> {
        self.layers.get(layer_idx)
    }

    /// Append key/value tensors for `layer_idx`, updating tracked sequence length.
    pub fn append(&mut self, layer_idx: usize, chunk: KvCacheChunk) -> Result<()> {
        let chunk_len = chunk.seq_len();
        let current_len = self
            .layers
            .get(layer_idx)
            .map(|kv| kv.seq_len())
            .unwrap_or(0);
        let new_len = current_len + chunk_len;
        match self.seq_len {
            None => {
                self.seq_len = Some(new_len);
            }
            Some(prev) => {
                ensure!(
                    new_len >= prev,
                    "cache seq_len decreased for layer {layer_idx}: {new_len} < {prev}"
                );
                if new_len > prev {
                    self.seq_len = Some(new_len);
                }
            }
        }
        self.layers.append_chunk(layer_idx, chunk)?;
        Ok(())
    }

    /// Returns the total number of layers being tracked (including empty ones).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Report the most recent total sequence length cached, if any.
    pub fn seq_len(&self) -> Option<usize> {
        self.seq_len
    }

    /// Clears all cached state.
    pub fn clear(&mut self) {
        self.layers.clear();
        self.seq_len = None;
    }

    /// Ensure the underlying cache tracks at least `total_layers` entries.
    pub fn ensure_layers(&mut self, total_layers: usize) {
        self.layers.ensure_layers(total_layers);
    }

    /// Borrow the underlying layer cache (e.g., for read-only access).
    pub fn layers(&self) -> &LayerKvCache {
        &self.layers
    }

    /// Mutable reference to the underlying layer cache.
    pub fn layers_mut(&mut self) -> &mut LayerKvCache {
        &mut self.layers
    }

    /// Returns a guard that automatically clears the cache when it falls out of scope.
    pub fn prompt_guard(&mut self) -> PromptCacheGuard<'_> {
        PromptCacheGuard::new(self)
    }

    /// Returns a guard that clears the cache and executes the provided reset hook when dropped.
    pub fn prompt_guard_with_reset<'a, F>(&'a mut self, reset: F) -> PromptCacheGuard<'a>
    where
        F: FnOnce() + 'a,
    {
        PromptCacheGuard::with_rope_reset(self, reset)
    }
}
