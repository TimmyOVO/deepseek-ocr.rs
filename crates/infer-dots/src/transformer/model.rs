use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::rms_norm;
use deepseek_ocr_core::{
    cache::{DynamicCache, PromptCacheGuard},
    tensor::gather_token_embeddings,
};

use crate::config::DotsOcrTextConfig;

use super::{block::Qwen2Block, rope::RopeCache};

#[derive(Debug)]
pub struct LanguageModelOutput {
    pub hidden_states: Tensor,
    pub logits: Tensor,
}

pub struct Qwen2LanguageModel {
    cfg: Arc<DotsOcrTextConfig>,
    blocks: Vec<Qwen2Block>,
    token_embedding: Tensor,
    final_norm: Tensor,
    lm_head: Tensor,
    rope: Mutex<RopeCache>,
}

impl Qwen2LanguageModel {
    pub fn load(cfg: Arc<DotsOcrTextConfig>, vb: &candle_nn::VarBuilder) -> Result<Self> {
        let model_vb = vb.pp("model");
        let token_embedding = model_vb
            .pp("embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")
            .context("missing embed_tokens weight")?;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer_vb = model_vb.pp(&format!("layers.{idx}"));
            blocks.push(Qwen2Block::load(cfg.as_ref(), &layer_vb)?);
        }
        let final_norm = model_vb
            .pp("norm")
            .get(cfg.hidden_size, "weight")
            .context("missing final norm weight")?;
        let lm_head = if cfg.tie_word_embeddings {
            token_embedding.clone()
        } else {
            vb.pp("lm_head")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")
                .context("missing lm_head weight")?
        };
        let rope_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rope = RopeCache::new(vb.device(), vb.dtype(), rope_dim, cfg.rope_theta)?;
        Ok(Self {
            cfg,
            blocks,
            token_embedding,
            final_norm,
            lm_head,
            rope: Mutex::new(rope),
        })
    }

    pub fn config(&self) -> &DotsOcrTextConfig {
        self.cfg.as_ref()
    }

    pub fn embed_tokens(&self) -> &Tensor {
        &self.token_embedding
    }

    pub fn new_cache(&self) -> DynamicCache {
        DynamicCache::with_num_layers(self.blocks.len())
    }

    pub fn prompt_guard<'a>(&'a self, cache: &'a mut DynamicCache) -> PromptCacheGuard<'a> {
        cache.prompt_guard()
    }

    pub fn forward(
        &self,
        input_ids: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        mut cache: Option<&mut DynamicCache>,
        use_cache: bool,
    ) -> Result<LanguageModelOutput> {
        ensure!(
            input_ids.is_some() ^ inputs_embeds.is_some(),
            "provide exactly one of input_ids or inputs_embeds"
        );
        ensure!(
            !use_cache || cache.is_some(),
            "use_cache=true requires mutable cache"
        );
        let embeddings = match inputs_embeds {
            Some(t) => t.clone(),
            None => {
                let ids = input_ids.expect("input_ids validated");
                let ids = if ids.dtype() == DType::I64 {
                    ids.clone()
                } else {
                    ids.to_dtype(DType::I64)?
                };
                gather_token_embeddings(&self.token_embedding, &ids)?
            }
        };
        let (batch, seq_len, _) = embeddings.shape().dims3()?;
        let past_len = cache.as_ref().and_then(|c| c.seq_len()).unwrap_or(0);
        let total_len = past_len + seq_len;
        let attn_mask = match attention_mask {
            Some(mask) => {
                let dtype_match = if mask.dtype() == embeddings.dtype() {
                    mask.clone()
                } else {
                    mask.to_dtype(embeddings.dtype())?
                };
                dtype_match
            }
            None => build_causal_mask(
                batch,
                seq_len,
                total_len,
                past_len,
                embeddings.dtype(),
                embeddings.device(),
            )?,
        };
        let pos_ids = match position_ids {
            Some(ids) => normalize_position_ids(ids, embeddings.device())?,
            None => default_position_ids(batch, seq_len, past_len, embeddings.device())?,
        };
        let (cos, sin) = {
            let mut rope = self.rope.lock().expect("rope cache mutex poisoned");
            rope.ensure_len(total_len)?;
            rope.select(batch, seq_len, Some(&pos_ids))?
        };
        if let Some(cache) = cache.as_ref() {
            ensure!(
                cache.num_layers() == 0 || cache.num_layers() >= self.blocks.len(),
                "cache tracks {} layers but model expects {}",
                cache.num_layers(),
                self.blocks.len()
            );
        }
        if let Some(cache_slot) = cache.as_mut() {
            (**cache_slot).ensure_layers(self.blocks.len());
        }
        let mut hidden = embeddings;
        for (idx, block) in self.blocks.iter().enumerate() {
            let (next, present) = {
                let past_entry = cache.as_ref().and_then(|c| c.get(idx));
                block.forward(&hidden, &cos, &sin, Some(&attn_mask), past_entry, use_cache)?
            };
            hidden = next;
            if let (Some(cache_slot), Some(chunk)) = (cache.as_mut(), present) {
                (**cache_slot).append(idx, chunk)?;
            }
        }
        let normed = rms_norm(&hidden, &self.final_norm, self.cfg.rms_norm_eps as f32)
            .context("final rms norm failed")?;
        let (batch, seq_len, hidden_size) = normed.shape().dims3()?;
        let logits = normed
            .reshape((batch * seq_len, hidden_size))?
            .matmul(&self.lm_head.transpose(0, 1)?)?
            .reshape((batch, seq_len, self.cfg.vocab_size))?;
        Ok(LanguageModelOutput {
            hidden_states: normed,
            logits,
        })
    }
}

fn build_causal_mask(
    batch: usize,
    seq_len: usize,
    total_len: usize,
    past_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut data = vec![0f32; batch * seq_len * total_len];
    for b in 0..batch {
        for q in 0..seq_len {
            let current = past_len + q;
            for k in (current + 1)..total_len {
                let idx = b * seq_len * total_len + q * total_len + k;
                data[idx] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(data, (batch, 1, seq_len, total_len), device)?.to_dtype(dtype)?)
}

fn default_position_ids(
    batch: usize,
    seq_len: usize,
    past_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let start = past_len as i64;
    let end = start + seq_len as i64;
    Ok(Tensor::arange(start, end, device)?
        .reshape((1, seq_len))?
        .expand((batch, seq_len))?
        .to_dtype(DType::I64)?)
}

fn normalize_position_ids(ids: &Tensor, device: &Device) -> Result<Tensor> {
    if ids.device().location() == device.location() && ids.dtype() == DType::I64 {
        return Ok(ids.clone());
    }
    Ok(ids.to_device(device)?.to_dtype(DType::I64)?)
}
