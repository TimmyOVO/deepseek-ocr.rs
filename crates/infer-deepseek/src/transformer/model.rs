use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, IndexOp, Tensor, quantized::QMatMul};
use candle_nn::ops::rms_norm;
use deepseek_ocr_core::tensor::gather_token_embeddings;

use crate::{
    config::DeepseekV2Config,
    quant_snapshot::QuantizedSnapshot,
    quantization::run_quantized_matmul,
    transformer::{
        cache::{DynamicCache, PromptCacheGuard},
        decoder::TransformerDecoder,
        weights::{DeepseekLanguageModelWeights, TransformerWeights},
    },
};

/// Output of a language-model forward pass.
#[derive(Debug)]
pub struct LanguageModelOutput {
    pub hidden_states: Tensor,
    pub logits: Tensor,
    pub aux_loss: Option<Tensor>,
}

/// Candle-backed implementation of the DeepSeek text decoder stack.
///
/// Responsibilities covered here:
/// - token embedding lookup (or accepting caller-provided embeddings)
/// - rotary-position-aware transformer decoding with optional KV caching
/// - final RMSNorm + vocab projection to produce logits
pub struct DeepseekLanguageModel {
    cfg: Arc<DeepseekV2Config>,
    decoder: TransformerDecoder,
    transformer_weights: Arc<TransformerWeights>,
    token_embedding: Tensor,
    final_layernorm: Tensor,
    lm_head_weight: Option<Tensor>,
    lm_head_q: Option<Arc<QMatMul>>,
    lm_out_dim: usize,
    lm_in_dim: usize,
    lm_head_label: String,
}

impl DeepseekLanguageModel {
    /// Load language-model weights from a [`VarBuilder`]-compatible source.
    pub fn load(cfg: Arc<DeepseekV2Config>, vb: &candle_nn::VarBuilder) -> Result<Self> {
        Self::load_with_snapshot(cfg, vb, None)
    }

    pub fn load_with_snapshot(
        cfg: Arc<DeepseekV2Config>,
        vb: &candle_nn::VarBuilder,
        snapshot: Option<&QuantizedSnapshot>,
    ) -> Result<Self> {
        let weights = DeepseekLanguageModelWeights::load(&cfg, vb, snapshot)?;
        Ok(Self::from_weights(cfg, weights))
    }

    /// Construct the language model from pre-loaded weight tensors.
    pub fn from_weights(cfg: Arc<DeepseekV2Config>, weights: DeepseekLanguageModelWeights) -> Self {
        let transformer = Arc::new(weights.transformer);
        let use_flash_attention = cfg
            .attn_implementation
            .as_deref()
            .map(|s| s.eq_ignore_ascii_case("flash_attention_2"))
            .unwrap_or(false);
        let decoder = TransformerDecoder::new(
            Arc::clone(&cfg),
            Arc::clone(&transformer),
            use_flash_attention,
        );
        Self {
            cfg,
            decoder,
            transformer_weights: transformer,
            token_embedding: weights.token_embedding,
            final_layernorm: weights.final_layernorm.weight,
            lm_head_weight: weights.lm_head_weight,
            lm_head_q: weights.lm_head_q,
            lm_out_dim: weights.lm_out_dim,
            lm_in_dim: weights.lm_in_dim,
            lm_head_label: weights.lm_head_label,
        }
    }

    pub fn config(&self) -> &DeepseekV2Config {
        self.cfg.as_ref()
    }

    pub fn transformer_weights(&self) -> &TransformerWeights {
        self.transformer_weights.as_ref()
    }

    #[doc(hidden)]
    pub fn transformer_weights_arc(&self) -> Arc<TransformerWeights> {
        Arc::clone(&self.transformer_weights)
    }

    pub fn flash_attention_enabled(&self) -> bool {
        self.decoder.flash_attention_enabled()
    }

    /// Lookup token embeddings for the provided input ids.
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        let ids = if input_ids.dtype() == DType::I64 {
            input_ids.clone()
        } else {
            input_ids.to_dtype(DType::I64)?
        };
        gather_token_embeddings(&self.token_embedding, &ids)
    }

    pub fn token_embedding_for_id(&self, token_id: usize) -> Result<Tensor> {
        let (vocab, _) = self.token_embedding.shape().dims2()?;
        ensure!(
            token_id < vocab,
            "token id {token_id} out of bounds for vocab size {vocab}"
        );
        Ok(self.token_embedding.i(token_id)?)
    }

    pub fn prompt_guard<'a>(&'a self, cache: &'a mut DynamicCache) -> PromptCacheGuard<'a> {
        self.decoder.prompt_guard(cache)
    }

    /// Forward pass through the language stack.
    ///
    /// Provide either `input_ids` **or** `inputs_embeds`. When `input_ids` are supplied, token
    /// embeddings are gathered using the stored embedding matrix. If `position_ids` are omitted,
    /// monotonically increasing positions are synthesized based on the current cache length.
    pub fn forward(
        &self,
        input_ids: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        cache: Option<&mut DynamicCache>,
        use_cache: bool,
    ) -> Result<LanguageModelOutput> {
        ensure!(
            input_ids.is_some() ^ inputs_embeds.is_some(),
            "provide exactly one of input_ids or inputs_embeds"
        );
        ensure!(
            !use_cache || cache.is_some(),
            "use_cache=true requires a mutable DynamicCache"
        );

        let past_len = cache.as_ref().and_then(|c| c.seq_len()).unwrap_or(0);
        let embeds = match inputs_embeds {
            Some(t) => t.clone(),
            None => {
                let ids = input_ids.expect("input_ids validity checked above");
                let ids = if ids.dtype() == DType::I64 {
                    ids.clone()
                } else {
                    ids.to_dtype(DType::I64)?
                };
                gather_token_embeddings(&self.token_embedding, &ids)?
            }
        };

        let (batch, seq_len, _) = embeds.shape().dims3()?;

        let position_buf: Option<Tensor> = if position_ids.is_some() {
            None
        } else {
            let device = embeds.device();
            let start = past_len as i64;
            let end = start + seq_len as i64;
            Some(
                Tensor::arange(start, end, device)?
                    .reshape((1, seq_len))?
                    .expand((batch, seq_len))?
                    .contiguous()?,
            )
        };
        let position_ids_ref: Option<&Tensor> = match position_ids {
            Some(ids) => Some(ids),
            None => position_buf.as_ref().map(|t| t as &Tensor),
        };

        let decoder_out =
            self.decoder
                .forward(&embeds, attention_mask, position_ids_ref, cache, use_cache)?;

        let normed = rms_norm(
            &decoder_out.hidden_states,
            &self.final_layernorm,
            self.cfg.rms_norm_eps as f32,
        )?;
        let (b, s, h) = normed.shape().dims3()?;
        ensure!(
            h == self.lm_in_dim,
            "lm_head expects hidden {}, got {}",
            self.lm_in_dim,
            h
        );
        let flat = normed.reshape((b * s, h))?.contiguous()?;
        let logits = if let Some(qm) = &self.lm_head_q {
            run_quantized_matmul(&self.lm_head_label, qm, &flat)?
        } else {
            let w = self
                .lm_head_weight
                .as_ref()
                .context("lm_head float weight missing for non-quantized path")?;
            flat.matmul(&w.transpose(0, 1)?)?
        };
        let logits = logits.reshape((b, s, self.lm_out_dim))?;

        Ok(LanguageModelOutput {
            hidden_states: normed,
            logits,
            aux_loss: decoder_out.aux_loss,
        })
    }
}
