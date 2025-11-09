use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::{rms_norm, silu};

use crate::config::PaddleOcrVlConfig;

use super::{
    attention::{AttentionContext, attention_forward},
    cache::{KvCacheChunk, KvCacheEntry},
    ops::apply_linear,
    weights::{ErnieDecoderLayerWeights, ErnieMlpWeights},
};

pub struct LayerOutput {
    pub hidden_states: Tensor,
    pub present_key_value: Option<KvCacheChunk>,
}

pub fn decoder_layer_forward(
    cfg: &PaddleOcrVlConfig,
    layer: &ErnieDecoderLayerWeights,
    hidden_states: &Tensor,
    attn_bias: Option<&Tensor>,
    rotary_ctx: &AttentionContext<'_>,
    cos: &Tensor,
    sin: &Tensor,
    past: Option<&KvCacheEntry>,
    use_cache: bool,
) -> Result<LayerOutput> {
    let residual = hidden_states;
    let normed = rms_norm(
        hidden_states,
        &layer.input_layernorm,
        cfg.rms_norm_eps as f32,
    )
    .context("input rms norm failed")?;
    let (attn_out, present) = attention_forward(
        rotary_ctx,
        &normed,
        &layer.attention,
        cos,
        sin,
        attn_bias,
        past,
        use_cache,
    )?;
    let hidden_states = residual
        .add(&attn_out)
        .context("attention residual add failed")?;

    let residual = &hidden_states;
    let normed = rms_norm(
        residual,
        &layer.post_attention_layernorm,
        cfg.rms_norm_eps as f32,
    )
    .context("post-attention rms norm failed")?;
    let mlp_out = mlp_forward(&normed, &layer.mlp, cfg).context("mlp forward failed")?;
    let hidden_states = residual.add(&mlp_out).context("mlp residual add failed")?;

    Ok(LayerOutput {
        hidden_states,
        present_key_value: present,
    })
}

fn mlp_forward(input: &Tensor, mlp: &ErnieMlpWeights, cfg: &PaddleOcrVlConfig) -> Result<Tensor> {
    let gate = apply_linear(input, &mlp.gate_proj)?;
    let up = apply_linear(input, &mlp.up_proj)?;
    let activated = match cfg.hidden_act.as_str() {
        "silu" | "swiglu" | "silu_glu" => silu(&gate)?,
        other => anyhow::bail!("unsupported activation: {other}"),
    };
    let fused = activated.mul(&up)?;
    apply_linear(&fused, &mlp.down_proj)
}

pub fn build_attention_bias(
    pad_mask: Option<&Tensor>,
    batch: usize,
    q_len: usize,
    k_len: usize,
    past_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Option<Tensor>> {
    let mut bias: Option<Tensor> = None;
    if past_len == 0 && q_len == k_len && q_len > 1 {
        let rows = Tensor::arange(0i64, q_len as i64, device)?.reshape((q_len, 1))?;
        let cols = Tensor::arange(0i64, k_len as i64, device)?.reshape((1, k_len))?;
        let mask = cols.broadcast_gt(&rows)?;
        let mask = mask.to_dtype(dtype)?;
        let fill =
            Tensor::full(mask_fill_value(dtype), mask.shape().clone(), device)?.to_dtype(dtype)?;
        let causal = mask.mul(&fill)?;
        let causal = causal.reshape((1, 1, q_len, k_len))?;
        let causal = causal.expand((batch, 1, q_len, k_len))?;
        bias = Some(causal);
    }

    if let Some(mask) = pad_mask {
        let (mask_batch, mask_len) = mask.shape().dims2()?;
        anyhow::ensure!(
            mask_batch == batch,
            "padding mask batch {} does not match input batch {}",
            mask_batch,
            batch
        );
        anyhow::ensure!(
            mask_len == k_len,
            "padding mask seq {} does not match key length {}",
            mask_len,
            k_len
        );
        let mask = if mask.dtype() == dtype {
            mask.clone()
        } else {
            mask.to_dtype(dtype)?
        };
        let ones = Tensor::full(1f32, (batch, k_len), device)?.to_dtype(dtype)?;
        let inverted = ones.sub(&mask)?;
        let inverted = inverted.reshape((batch, 1, 1, k_len))?;
        let fill = Tensor::full(mask_fill_value(dtype), inverted.shape().clone(), device)?
            .to_dtype(dtype)?;
        let pad_bias = inverted.mul(&fill)?;
        bias = Some(if let Some(existing) = bias {
            existing.broadcast_add(&pad_bias)?
        } else {
            pad_bias
        });
    }

    Ok(bias)
}

fn mask_fill_value(dtype: DType) -> f32 {
    match dtype {
        DType::F16 | DType::BF16 => -1e4f32,
        _ => -1e9f32,
    }
}
