use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Tensor};
use deepseek_ocr_core::{
    benchmark::Timer,
    cache::{DynamicCache, PromptCacheGuard},
    tensor::gather_token_embeddings,
};

use crate::config::GlmOcrTextConfig;

use super::{
    attention::build_attention_bias,
    block::decoder_layer_forward,
    rope::GlmTextRotaryEmbedding,
    weights::{GlmTextLayerWeights, LinearWeights},
};

pub struct DecoderOutput {
    pub logits: Tensor,
}

pub struct GlmTextDecoder {
    cfg: Arc<GlmOcrTextConfig>,
    embed_tokens: Tensor,
    layers: Vec<GlmTextLayerWeights>,
    norm: Tensor,
    lm_head: LinearWeights,
    rotary: GlmTextRotaryEmbedding,
}

impl GlmTextDecoder {
    pub fn load(cfg: Arc<GlmOcrTextConfig>, vb: &candle_nn::VarBuilder) -> Result<Self> {
        let model_vb = vb.pp("model").pp("language_model");
        let embed_tokens = model_vb
            .pp("embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")
            .context("missing model.language_model.embed_tokens.weight")?
            .contiguous()?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            let layer_vb = model_vb.pp(format!("layers.{idx}"));
            layers.push(GlmTextLayerWeights::load(
                &layer_vb,
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads.max(1),
                cfg.head_dim,
                cfg.attention_bias,
            )?);
        }

        let norm = model_vb
            .pp("norm")
            .get(cfg.hidden_size, "weight")
            .context("missing model.language_model.norm.weight")?
            .contiguous()?;
        let lm_head = LinearWeights::load(
            vb.pp("lm_head"),
            cfg.vocab_size,
            cfg.hidden_size,
            false,
        )?;
        let rotary = GlmTextRotaryEmbedding::new(Arc::clone(&cfg))?;

        Ok(Self {
            cfg,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
        })
    }

    pub fn embed_tokens(&self) -> &Tensor {
        &self.embed_tokens
    }

    pub fn new_cache(&self) -> DynamicCache {
        DynamicCache::with_num_layers(self.layers.len())
    }

    pub fn prompt_guard<'a>(&'a self, cache: &'a mut DynamicCache) -> PromptCacheGuard<'a> {
        cache.prompt_guard()
    }

    #[allow(clippy::too_many_arguments)]
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
            "use_cache=true requires mutable cache"
        );

        let embeddings = match inputs_embeds {
            Some(t) => t.clone(),
            None => {
                let ids = input_ids.expect("validated input ids");
                let ids = if ids.dtype() == DType::I64 {
                    ids.clone()
                } else {
                    ids.to_dtype(DType::I64)?
                };
                gather_token_embeddings(&self.embed_tokens, &ids)?
            }
        };

        let (batch, seq_len, _) = embeddings.shape().dims3()?;
        let past_len = cache.as_ref().and_then(|c| c.seq_len()).unwrap_or(0);
        let total_k_len = past_len + seq_len;

        let attn_bias = if attention_mask.is_none() && seq_len == 1 {
            None
        } else {
            Some(build_attention_bias(
                attention_mask,
                batch,
                seq_len,
                total_k_len,
                past_len,
                embeddings.dtype(),
                embeddings.device(),
            )?)
        };

        let pos_ids = match position_ids {
            Some(ids) => normalize_position_ids(ids, embeddings.device(), batch, seq_len)?,
            None => default_position_ids(batch, seq_len, past_len, embeddings.device())?,
        };
        let (cos, sin) = self.rotary.cos_sin(&pos_ids, embeddings.dtype())?;

        if let Some(existing) = cache.as_ref() {
            ensure!(
                existing.num_layers() == 0 || existing.num_layers() >= self.layers.len(),
                "cache layers {} smaller than model layers {}",
                existing.num_layers(),
                self.layers.len()
            );
        }
        if let Some(existing) = cache.as_mut() {
            existing.ensure_layers(self.layers.len());
        }

        let mut hidden = embeddings;
        for (idx, layer) in self.layers.iter().enumerate() {
            let past = cache.as_deref().and_then(|c| c.get(idx));
            let layer_timer = Timer::new("text.layer.forward");
            let out = decoder_layer_forward(
                self.cfg.as_ref(),
                layer,
                &hidden,
                attn_bias.as_ref(),
                &cos,
                &sin,
                past,
                use_cache,
            )?;
            layer_timer.finish(|_| {});
            hidden = out.hidden_states;
            if let Some(chunk) = out.present_key_value
                && let Some(cache_mut) = cache.as_deref_mut()
            {
                let append_timer = Timer::new("text.layer.cache_append");
                cache_mut.append(idx, chunk)?;
                append_timer.finish(|_| {});
            }
        }

        let hidden = rms_norm_precise(&hidden, &self.norm, self.cfg.rms_norm_eps)?;
        let (batch, seq, hidden_size) = hidden.shape().dims3()?;
        let flat = hidden.reshape((batch * seq, hidden_size))?;
        let logits = self
            .lm_head
            .matmul_2d(&flat)?
            .reshape((batch, seq, self.cfg.vocab_size))?;

        Ok(DecoderOutput {
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
    let stacked = Tensor::stack(&[base.clone(), base.clone(), base], 0)?;
    Ok(stacked.to_dtype(DType::I64)?)
}

fn normalize_position_ids(
    ids: &Tensor,
    device: &candle_core::Device,
    batch: usize,
    seq_len: usize,
) -> Result<Tensor> {
    let ids = if ids.device().location() == device.location() {
        ids.clone()
    } else {
        ids.to_device(device)?
    };
    match ids.rank() {
        3 => {
            let (axes, b, s) = ids.shape().dims3()?;
            ensure!(axes == 3 && b == batch && s == seq_len, "position_ids shape mismatch");
            Ok(ids.to_dtype(DType::I64)?)
        }
        2 => {
            let (b, s) = ids.shape().dims2()?;
            ensure!(b == batch && s == seq_len, "position_ids shape mismatch");
            let expanded = ids.unsqueeze(0)?.expand((3, batch, seq_len))?.contiguous()?;
            Ok(expanded.to_dtype(DType::I64)?)
        }
        other => anyhow::bail!("position_ids rank must be 2 or 3, got {other}"),
    }
}

fn rms_norm_precise(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = input.dtype();
    let x = input.to_dtype(DType::F32)?;
    let hidden = x.dim(candle_core::shape::D::Minus1)?;
    let variance =
        (x.sqr()?.sum_keepdim(candle_core::shape::D::Minus1)? / hidden as f64)?;
    let inv = (variance + eps)?.sqrt()?.recip()?;
    let normed = x.broadcast_mul(&inv)?;
    let weight = if weight.dtype() == DType::F32 {
        weight.clone()
    } else {
        weight.to_dtype(DType::F32)?
    };
    Ok(normed.broadcast_mul(&weight)?.to_dtype(dtype)?)
}
