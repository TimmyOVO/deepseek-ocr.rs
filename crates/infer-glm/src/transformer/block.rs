use anyhow::{Context, Result};
use candle_core::Tensor;

use crate::config::GlmOcrTextConfig;

use super::{
    attention::{TextAttentionContext, attention_forward},
    weights::{GlmTextLayerWeights, GlmTextMlpWeights},
};

pub struct LayerOutput {
    pub hidden_states: Tensor,
    pub present_key_value: Option<deepseek_ocr_core::cache::KvCacheChunk>,
}

#[allow(clippy::too_many_arguments)]
pub fn decoder_layer_forward(
    cfg: &GlmOcrTextConfig,
    layer: &GlmTextLayerWeights,
    hidden_states: &Tensor,
    attn_bias: Option<&Tensor>,
    cos: &Tensor,
    sin: &Tensor,
    past: Option<&deepseek_ocr_core::cache::KvCacheEntry>,
    use_cache: bool,
) -> Result<LayerOutput> {
    let residual = hidden_states;
    let normed = rms_norm_precise(hidden_states, &layer.input_layernorm, cfg.rms_norm_eps)
        .context("input rms norm failed")?;

    let attn_ctx = TextAttentionContext {
        num_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads.max(1),
        head_dim: cfg.head_dim,
        scaling: cfg.head_dim as f64,
    };
    let (attn_out, present) = attention_forward(
        &attn_ctx,
        &normed,
        &layer.attention,
        cos,
        sin,
        attn_bias,
        past,
        use_cache,
    )?;
    let attn_out = rms_norm_precise(&attn_out, &layer.post_self_attn_layernorm, cfg.rms_norm_eps)
        .context("post-self-attn rms norm failed")?;
    let hidden_states = residual
        .add(&attn_out)
        .context("attention residual add failed")?;

    let residual = &hidden_states;
    let normed = rms_norm_precise(residual, &layer.post_attention_layernorm, cfg.rms_norm_eps)
        .context("post-attention rms norm failed")?;
    let mlp_out = mlp_forward(&normed, &layer.mlp).context("mlp forward failed")?;
    let mlp_out = rms_norm_precise(&mlp_out, &layer.post_mlp_layernorm, cfg.rms_norm_eps)
        .context("post-mlp rms norm failed")?;
    let hidden_states = residual.add(&mlp_out).context("mlp residual add failed")?;

    Ok(LayerOutput {
        hidden_states,
        present_key_value: present,
    })
}

fn mlp_forward(input: &Tensor, mlp: &GlmTextMlpWeights) -> Result<Tensor> {
    let gate_up = mlp.gate_up_proj.forward_3d(input)?;
    let hidden = gate_up.dim(candle_core::shape::D::Minus1)?;
    let half = hidden / 2;
    let gate = gate_up.narrow(candle_core::shape::D::Minus1, 0, half)?;
    let up = gate_up.narrow(candle_core::shape::D::Minus1, half, half)?;
    let fused = up.broadcast_mul(&gate.silu()?)?;
    mlp.down_proj.forward_3d(&fused)
}

fn rms_norm_precise(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = input.dtype();
    let x = input.to_dtype(candle_core::DType::F32)?;
    let hidden = x.dim(candle_core::shape::D::Minus1)?;
    let variance = (x.sqr()?.sum_keepdim(candle_core::shape::D::Minus1)? / hidden as f64)?;
    let inv = (variance + eps)?.sqrt()?.recip()?;
    let normed = x.broadcast_mul(&inv)?;
    let weight = if weight.dtype() == candle_core::DType::F32 {
        weight.clone()
    } else {
        weight.to_dtype(candle_core::DType::F32)?
    };
    Ok(normed.broadcast_mul(&weight)?.to_dtype(dtype)?)
}
