use crate::{
    config::DeepseekV2Config,
    transformer::{
        block::{TransformerBlock, build_attention_bias},
        cache::{DynamicCache, PromptCacheGuard},
        rope::RopeCache,
        weights::TransformerWeights,
    },
};
use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};
use std::{cell::RefCell, sync::Arc};

/// Runs the stacked transformer decoder layers, handling optional KV cache reuse.
pub struct TransformerDecoder {
    cfg: Arc<DeepseekV2Config>,
    weights: Arc<TransformerWeights>,
    rope_cache: RefCell<Option<RopeCache>>,
    use_flash_attention: bool,
}

fn parse_layer_slice(spec: &str) -> Option<(usize, Option<usize>)> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(idx) = trimmed.find("..") {
        let start_str = &trimmed[..idx];
        let end_str = &trimmed[idx + 2..];
        let start = start_str.trim().parse::<usize>().ok()?;
        let end = if end_str.trim().is_empty() {
            None
        } else {
            Some(end_str.trim().parse::<usize>().ok()?)
        };
        Some((start, end))
    } else {
        let value = trimmed.parse::<usize>().ok()?;
        Some((value, Some(value + 1)))
    }
}

#[derive(Debug)]
pub struct DecoderOutput {
    pub hidden_states: Tensor,
    pub aux_loss: Option<Tensor>,
}

impl TransformerDecoder {
    pub fn new(
        cfg: Arc<DeepseekV2Config>,
        weights: Arc<TransformerWeights>,
        use_flash_attention: bool,
    ) -> Self {
        Self {
            cfg,
            weights,
            rope_cache: RefCell::new(None),
            use_flash_attention,
        }
    }

    pub fn flash_attention_enabled(&self) -> bool {
        self.use_flash_attention
    }

    /// Drops any cached RoPE tables so the next forward restarts from position zero.
    pub fn reset_rope_cache(&self) {
        self.rope_cache.borrow_mut().take();
        #[cfg(feature = "memlog")]
        deepseek_ocr_core::memlog::set_rope(0);
    }

    /// Returns a guard that clears both the KV cache and the decoder's RoPE tables when dropped.
    pub fn prompt_guard<'b>(&'b self, cache: &'b mut DynamicCache) -> PromptCacheGuard<'b> {
        cache.prompt_guard_with_reset(|| self.reset_rope_cache())
    }

    /// Forward pass across all decoder layers.
    ///
    /// When `use_cache` is true, a mutable [`DynamicCache`] must be supplied. It will be updated
    /// in-place with newly appended KV entries while being used as the source for `past_key_values`.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        mut cache: Option<&mut DynamicCache>,
        use_cache: bool,
    ) -> Result<DecoderOutput> {
        ensure!(
            !use_cache || cache.is_some(),
            "use_cache=true requires a mutable DynamicCache"
        );
        let past_len = cache.as_ref().and_then(|c| c.seq_len()).unwrap_or(0);
        let (batch, q_len, _) = hidden_states.shape().dims3()?;
        let dtype = hidden_states.dtype();
        let device = hidden_states.device();
        let k_len = past_len + q_len;

        let attn_bias =
            build_attention_bias(attention_mask, batch, q_len, k_len, past_len, dtype, device)?;

        let total_layers = self.weights.layers.len();
        let (layer_start, layer_end) = std::env::var("DEEPSEEK_OCR_LAYER_SLICE")
            .ok()
            .and_then(|spec| parse_layer_slice(&spec))
            .map(|(start, end_opt)| {
                let end = end_opt.unwrap_or(total_layers);
                (start.min(total_layers), end.min(total_layers))
            })
            .unwrap_or((0, total_layers));
        ensure!(
            layer_start < layer_end,
            "invalid layer slice: {}..{}",
            layer_start,
            layer_end
        );

        let head_dim = self.cfg.hidden_size / self.cfg.num_attention_heads;
        let rope_dim_cfg = self.cfg.qk_rope_head_dim.unwrap_or(head_dim);
        let rope_dim = if rope_dim_cfg == 0 {
            head_dim
        } else {
            rope_dim_cfg
        };
        let position_ids_local = if let Some(ids) = position_ids {
            Some(if ids.dtype() == DType::I64 {
                ids.clone()
            } else {
                ids.to_dtype(DType::I64)?
            })
        } else {
            let start = past_len as i64;
            let end = start + q_len as i64;
            Some(
                Tensor::arange(start, end, device)?
                    .reshape((1, q_len))?
                    .expand((batch, q_len))?
                    .contiguous()?
                    .to_dtype(DType::I64)?,
            )
        };
        let ids_for_rope = position_ids_local.as_ref();
        let mut rope_tensors: Option<(Tensor, Tensor)> = None;
        if layer_start < total_layers {
            if rope_dim > 0 {
                let mut rope_entry = self.rope_cache.borrow_mut();
                let needs_new = match rope_entry.as_ref() {
                    Some(cache) => !cache.matches(dtype, rope_dim, device),
                    None => true,
                };
                if needs_new {
                    *rope_entry = Some(RopeCache::new(device, dtype, rope_dim)?);
                }
                if let Some(cache) = rope_entry.as_mut() {
                    if let Some(ids) = ids_for_rope {
                        let want = if q_len == 0 {
                            past_len
                        } else {
                            let ids_cpu = ids.to_device(&candle_core::Device::Cpu)?;
                            let max_pos = ids_cpu.max_all()?.to_scalar::<i64>()? as usize;
                            (past_len + q_len).max(max_pos + 1)
                        };
                        cache.ensure_len(&self.cfg, want)?;
                        rope_tensors = Some(cache.select(batch, q_len, Some(ids))?);
                    }
                }
            } else {
                self.rope_cache.borrow_mut().take();
            }
        }

        let mut hidden = hidden_states.clone();
        let mut aux_loss: Option<Tensor> = None;
        if let Some(existing) = cache.as_ref() {
            ensure!(
                existing.num_layers() == 0 || existing.num_layers() >= total_layers,
                "provided cache tracks {} layers but model expects {}",
                existing.num_layers(),
                total_layers
            );
        }
        if let Some(existing) = cache.as_mut() {
            existing.ensure_layers(total_layers);
        }

        for (idx, layer_weights) in self.weights.layers[layer_start..layer_end]
            .iter()
            .enumerate()
            .map(|(i, w)| (layer_start + i, w))
        {
            let block = TransformerBlock::new(&self.cfg, layer_weights, self.use_flash_attention);
            let output = {
                let past = cache.as_deref().and_then(|cache| cache.get(idx));
                let rope_refs = rope_tensors.as_ref().map(|(cos, sin)| (cos, sin));
                block.forward(&hidden, attn_bias.as_ref(), rope_refs, past, use_cache)?
            };
            hidden = output.hidden_states;
            if let Some(present) = output.present_key_value {
                if let Some(cache) = cache.as_mut() {
                    cache.append(idx, present)?;
                }
            }
            if let Some(loss) = output.aux_loss {
                aux_loss = Some(match aux_loss {
                    Some(existing) => existing.add(&loss)?,
                    None => loss,
                });
            }
        }

        Ok(DecoderOutput {
            hidden_states: hidden,
            aux_loss,
        })
    }
}
