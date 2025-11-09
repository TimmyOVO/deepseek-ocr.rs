mod common;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use common::test_utils::{shared_language_config, shared_transformer_weights};
use deepseek_ocr_infer_deepseek::{
    config::DeepseekV2Config,
    transformer::{
        block::{TransformerBlock, lengths_to_padding_mask},
        rope::RopeCache,
    },
};

fn rope_for(
    cfg: &DeepseekV2Config,
    batch: usize,
    seq_len: usize,
    device: &Device,
    dtype: DType,
) -> Result<Option<(Tensor, Tensor)>> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    let rope_dim_cfg = cfg.qk_rope_head_dim.unwrap_or(head_dim);
    let rope_dim = if rope_dim_cfg == 0 {
        head_dim
    } else {
        rope_dim_cfg
    };
    if rope_dim == 0 {
        return Ok(None);
    }
    let mut cache = RopeCache::new(device, dtype, rope_dim)?;
    cache.ensure_len(cfg, seq_len)?;
    Ok(Some(cache.select(batch, seq_len, None)?))
}

#[test]
fn transformer_block_forward_shapes() -> Result<()> {
    let config = shared_language_config()?;
    let weights = shared_transformer_weights()?;
    let device = Device::Cpu;
    let block = TransformerBlock::new(config.as_ref(), &weights.layers[0], false);
    let hidden = Tensor::zeros((1, 4, config.hidden_size), DType::F32, &device)?;
    let rope = rope_for(config.as_ref(), 1, 4, &device, DType::F32)?;
    let output = block.forward(
        &hidden,
        None,
        rope.as_ref().map(|(c, s)| (c, s)),
        None,
        false,
    )?;
    assert_eq!(
        output.hidden_states.shape().dims3()?,
        (1, 4, config.hidden_size)
    );
    Ok(())
}

#[test]
fn lengths_to_padding_mask_builds_expected() -> Result<()> {
    let device = Device::Cpu;
    let mask = lengths_to_padding_mask(&[2, 4], 4, &device)?;
    let rows = mask.to_vec2::<f32>()?;
    assert_eq!(rows[0], vec![1.0, 1.0, 0.0, 0.0]);
    assert_eq!(rows[1], vec![1.0, 1.0, 1.0, 1.0]);
    Ok(())
}

#[test]
fn transformer_block_handles_padding_mask() -> Result<()> {
    let config = shared_language_config()?;
    let weights = shared_transformer_weights()?;
    let device = Device::Cpu;
    let block = TransformerBlock::new(config.as_ref(), &weights.layers[0], false);
    let hidden = Tensor::zeros((1, 4, config.hidden_size), DType::F32, &device)?;
    let mask = lengths_to_padding_mask(&[2], 4, &device)?;
    let rope = rope_for(config.as_ref(), 1, 4, &device, DType::F32)?;
    let output = block.forward(
        &hidden,
        Some(&mask),
        rope.as_ref().map(|(c, s)| (c, s)),
        None,
        false,
    )?;
    assert_eq!(
        output.hidden_states.shape().dims3()?,
        (1, 4, config.hidden_size)
    );
    Ok(())
}
