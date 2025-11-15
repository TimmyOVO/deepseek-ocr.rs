use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use candle_core::{DType, Tensor};
use candle_nn::ops::rms_norm;
use deepseek_ocr_core::tensor::gather_token_embeddings;

use crate::config::PaddleOcrVlConfig;

use super::{
    attention::{AttentionContext, supports_flash_attention},
    block::{build_attention_bias, decoder_layer_forward},
    cache::{DynamicCache, PromptCacheGuard},
    rope::ErnieRotaryEmbedding,
    weights::{ErnieModelWeights, LinearWeights},
};

pub struct DecoderOutput {
    pub hidden_states: Tensor,
    pub logits: Tensor,
}

pub struct ErnieDecoder {
    cfg: Arc<PaddleOcrVlConfig>,
    weights: ErnieModelWeights,
    lm_head: LinearWeights,
    rotary: ErnieRotaryEmbedding,
}

impl ErnieDecoder {
    pub fn load(
        cfg: Arc<PaddleOcrVlConfig>,
        vb: &candle_nn::VarBuilder,
        snapshot_hits: Option<&mut crate::snapshot::SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let model_vb = vb.pp("model");
        let mut snapshot_hits = snapshot_hits;
        let weights = ErnieModelWeights::load(
            &model_vb,
            cfg.as_ref(),
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )
        .context("failed to load Ernie decoder weights")?;
        let lm_head = LinearWeights::load(
            vb.pp("lm_head"),
            cfg.vocab_size,
            cfg.hidden_size,
            false,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )
        .context("failed to load lm_head weights")?;
        Self::from_parts(cfg, weights, lm_head)
    }

    pub(crate) fn from_parts(
        cfg: Arc<PaddleOcrVlConfig>,
        weights: ErnieModelWeights,
        lm_head: LinearWeights,
    ) -> Result<Self> {
        let rotary = ErnieRotaryEmbedding::new(Arc::clone(&cfg))?;
        Ok(Self {
            cfg,
            weights,
            lm_head,
            rotary,
        })
    }

    pub fn config(&self) -> &PaddleOcrVlConfig {
        self.cfg.as_ref()
    }

    pub fn embed_tokens(&self) -> &Tensor {
        &self.weights.embed_tokens
    }

    pub fn layers(&self) -> &[super::weights::ErnieDecoderLayerWeights] {
        &self.weights.layers
    }

    pub fn final_norm(&self) -> &Tensor {
        &self.weights.final_norm
    }

    pub fn lm_head(&self) -> &LinearWeights {
        &self.lm_head
    }

    pub fn flash_attention_enabled(&self) -> bool {
        self.cfg.use_flash_attention
    }

    pub fn new_cache(&self) -> DynamicCache {
        DynamicCache::with_num_layers(self.weights.layers.len())
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
    ) -> Result<DecoderOutput> {
        ensure!(
            input_ids.is_some() ^ inputs_embeds.is_some(),
            "provide exactly one of input_ids or inputs_embeds"
        );
        ensure!(
            !use_cache || cache.is_some(),
            "use_cache=true requires a mutable DynamicCache"
        );
        let embeddings = match inputs_embeds {
            Some(t) => t.clone(),
            None => {
                let ids = input_ids.expect("input_ids validity checked above");
                let ids = if ids.dtype() == DType::I64 {
                    ids.clone()
                } else {
                    ids.to_dtype(DType::I64)?
                };
                gather_token_embeddings(&self.weights.embed_tokens, &ids)?
            }
        };

        let (batch, seq_len, _) = embeddings.shape().dims3()?;
        let dtype = embeddings.dtype();
        let device = embeddings.device();
        let past_len = cache.as_ref().and_then(|c| c.seq_len()).unwrap_or(0);
        let k_len = past_len + seq_len;

        let pad_mask = attention_mask;
        let flash_prefill = pad_mask.is_none()
            && past_len == 0
            && supports_flash_attention(self.cfg.as_ref(), device, dtype);
        let attn_bias = if flash_prefill {
            None
        } else {
            build_attention_bias(pad_mask, batch, seq_len, k_len, past_len, dtype, device)?
        };

        let position_ids = match position_ids {
            Some(ids) => normalize_position_ids(ids, device)?,
            None => default_position_ids(batch, seq_len, past_len, device)?,
        };
        let (cos, sin) = self.rotary.cos_sin(&position_ids, dtype)?;
        let attn_ctx = AttentionContext {
            cfg: self.cfg.as_ref(),
            rotary: &self.rotary,
        };

        let total_layers = self.weights.layers.len();
        if let Some(existing) = cache.as_ref() {
            ensure!(
                existing.num_layers() == 0 || existing.num_layers() >= total_layers,
                "provided cache tracks {} layers but decoder expects {}",
                existing.num_layers(),
                total_layers
            );
        }
        if let Some(existing) = cache.as_mut() {
            existing.ensure_layers(total_layers);
        }

        let mut hidden = embeddings;
        for (idx, layer) in self.weights.layers.iter().enumerate() {
            let past = cache.as_deref().and_then(|c| c.get(idx));
            let output = decoder_layer_forward(
                self.cfg.as_ref(),
                layer,
                &hidden,
                attn_bias.as_ref(),
                &attn_ctx,
                &cos,
                &sin,
                past,
                use_cache,
            )?;
            hidden = output.hidden_states;
            if let Some(chunk) = output.present_key_value {
                if let Some(cache_mut) = cache.as_deref_mut() {
                    cache_mut.append(idx, chunk)?;
                }
            }
        }

        let normed = rms_norm(
            &hidden,
            &self.weights.final_norm,
            self.cfg.rms_norm_eps as f32,
        )
        .context("final rms norm failed")?;
        let (batch, seq_len, hidden_size) = normed.shape().dims3()?;
        let flat = normed.reshape((batch * seq_len, hidden_size))?;
        let logits = self.lm_head.matmul_2d(&flat)?;
        let logits = logits.reshape((batch, seq_len, self.cfg.vocab_size))?;

        Ok(DecoderOutput {
            hidden_states: normed,
            logits,
        })
    }
}

fn default_position_ids(
    batch: usize,
    seq_len: usize,
    past_len: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let start = past_len as i64;
    let end = start + seq_len as i64;
    let base = Tensor::arange(start, end, device)?
        .reshape((1, seq_len))?
        .expand((batch, seq_len))?
        .contiguous()?;
    Ok(Tensor::stack(&[base.clone(), base.clone(), base], 0)?.to_dtype(DType::I64)?)
}

fn normalize_position_ids(ids: &Tensor, device: &candle_core::Device) -> Result<Tensor> {
    match ids.rank() {
        3 => {
            if ids.device().location() == device.location() {
                Ok(ids.clone())
            } else {
                Ok(ids.to_device(device)?)
            }
        }
        2 => {
            let (batch, seq) = ids.shape().dims2()?;
            let expanded = ids.unsqueeze(0)?.expand((3, batch, seq))?.contiguous()?;
            Ok(expanded)
        }
        other => Err(anyhow!("position ids must have rank 2 or 3, got {}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::{
        ErnieAttentionWeights, ErnieDecoderLayerWeights, ErnieMlpWeights, ErnieModelWeights,
        LinearWeights,
    };
    use anyhow::Result;
    use candle_core::{DType, Device};
    use serde_json::json;

    fn test_config() -> PaddleOcrVlConfig {
        let cfg_json = json!({
            "head_dim": 6,
            "hidden_size": 18,
            "intermediate_size": 24,
            "vocab_size": 32,
            "num_attention_heads": 3,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "rope_scaling": { "mrope_section": [1,1,1], "type": "default" },
            "hidden_act": "silu",
            "use_bias": false,
            "use_cache": true,
            "use_flash_attention": false,
            "vision_config": {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_channels": 3,
                "image_size": 32,
                "patch_size": 4,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "tokens_per_second": 1
            }
        });
        serde_json::from_value(cfg_json).expect("valid config json")
    }

    fn linear(out_dim: usize, in_dim: usize, device: &Device) -> Result<LinearWeights> {
        Ok(LinearWeights {
            weight: Some(Tensor::zeros((out_dim, in_dim), DType::F32, device)?),
            bias: None,
            qmatmul: None,
            out_dim,
            in_dim,
            label: format!("test.linear.{out_dim}x{in_dim}"),
        })
    }

    fn layer_weights(cfg: &PaddleOcrVlConfig, device: &Device) -> Result<ErnieDecoderLayerWeights> {
        let attn = ErnieAttentionWeights {
            q_proj: linear(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                device,
            )?,
            k_proj: linear(
                cfg.resolved_num_key_value_heads() * cfg.head_dim,
                cfg.hidden_size,
                device,
            )?,
            v_proj: linear(
                cfg.resolved_num_key_value_heads() * cfg.head_dim,
                cfg.hidden_size,
                device,
            )?,
            o_proj: linear(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                device,
            )?,
        };
        let mlp = ErnieMlpWeights {
            gate_proj: linear(cfg.intermediate_size, cfg.hidden_size, device)?,
            up_proj: linear(cfg.intermediate_size, cfg.hidden_size, device)?,
            down_proj: linear(cfg.hidden_size, cfg.intermediate_size, device)?,
        };
        let norm = Tensor::ones(cfg.hidden_size, DType::F32, device)?;
        Ok(ErnieDecoderLayerWeights {
            attention: attn,
            mlp,
            input_layernorm: norm.clone(),
            post_attention_layernorm: norm,
        })
    }

    #[test]
    fn decoder_forward_shapes() -> Result<()> {
        let cfg = Arc::new(test_config());
        let device = Device::Cpu;
        let embed = Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F32, &device)?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| layer_weights(cfg.as_ref(), &device))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = Tensor::ones(cfg.hidden_size, DType::F32, &device)?;
        let weights = ErnieModelWeights {
            embed_tokens: embed,
            layers,
            final_norm,
        };
        let lm_head = LinearWeights {
            weight: Some(Tensor::zeros(
                (cfg.vocab_size, cfg.hidden_size),
                DType::F32,
                device,
            )?),
            bias: None,
            qmatmul: None,
            out_dim: cfg.vocab_size,
            in_dim: cfg.hidden_size,
            label: "tests.lm_head.weight".to_string(),
        };
        let decoder = ErnieDecoder::from_parts(Arc::clone(&cfg), weights, lm_head)?;

        let input_ids = Tensor::zeros((2, 4), DType::I64, &device)?;
        let output = decoder.forward(Some(&input_ids), None, None, None, None, false)?;
        assert_eq!(output.hidden_states.shape().dims(), [2, 4, cfg.hidden_size]);
        assert_eq!(output.logits.shape().dims(), [2, 4, cfg.vocab_size]);

        let mut cache = decoder.new_cache();
        decoder.forward(Some(&input_ids), None, None, None, Some(&mut cache), true)?;
        let next = Tensor::zeros((2, 1), DType::I64, &device)?;
        decoder.forward(Some(&next), None, None, None, Some(&mut cache), true)?;
        Ok(())
    }
}
