mod common;

use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use common::test_utils::{shared_language_config, shared_transformer_weights};
use deepseek_ocr_infer_deepseek::{
    config::DeepseekV2Config,
    transformer::{cache::DynamicCache, decoder::TransformerDecoder, weights::TransformerWeights},
};

fn decoder_with_single_layer() -> Result<(Arc<DeepseekV2Config>, Arc<TransformerWeights>)> {
    let cfg = shared_language_config()?;
    let mut weights = (*shared_transformer_weights()?).clone();
    weights.layers.truncate(1);
    Ok((cfg, Arc::new(weights)))
}

#[test]
fn decoder_forward_smoke() -> Result<()> {
    let (cfg, weights) = decoder_with_single_layer()?;
    let device = Device::Cpu;
    let decoder = TransformerDecoder::new(Arc::clone(&cfg), Arc::clone(&weights), false);
    let hidden = Tensor::zeros((1, 4, cfg.hidden_size), DType::F32, &device)?;
    let output = decoder.forward(&hidden, None, None, None, false)?;
    assert_eq!(
        output.hidden_states.shape().dims3()?,
        (1, 4, cfg.hidden_size)
    );
    Ok(())
}

#[test]
fn decoder_requires_cache_when_use_cache_enabled() -> Result<()> {
    let (cfg, weights) = decoder_with_single_layer()?;
    let device = Device::Cpu;
    let decoder = TransformerDecoder::new(Arc::clone(&cfg), Arc::clone(&weights), false);
    let hidden = Tensor::zeros((1, 3, cfg.hidden_size), DType::F32, &device)?;
    let err = decoder
        .forward(&hidden, None, None, None, true)
        .expect_err("use_cache without cache must be rejected");
    assert!(err.to_string().contains("requires a mutable DynamicCache"));
    Ok(())
}

#[test]
fn decoder_updates_dynamic_cache_in_place() -> Result<()> {
    let (cfg, weights) = decoder_with_single_layer()?;
    let device = Device::Cpu;
    let decoder = TransformerDecoder::new(Arc::clone(&cfg), Arc::clone(&weights), false);
    let mut cache = DynamicCache::with_num_layers(weights.layers.len());

    let hidden = Tensor::zeros((1, 4, cfg.hidden_size), DType::F32, &device)?;
    decoder.forward(&hidden, None, None, Some(&mut cache), true)?;
    assert_eq!(cache.seq_len(), Some(4));

    let next_hidden = Tensor::zeros((1, 1, cfg.hidden_size), DType::F32, &device)?;
    decoder.forward(&next_hidden, None, None, Some(&mut cache), true)?;
    assert_eq!(cache.seq_len(), Some(5));
    Ok(())
}

#[test]
fn decoder_prompt_guard_clears_rope_and_cache() -> Result<()> {
    let (cfg, weights) = decoder_with_single_layer()?;
    let device = Device::Cpu;
    let decoder = TransformerDecoder::new(Arc::clone(&cfg), Arc::clone(&weights), false);
    let mut cache = DynamicCache::with_num_layers(weights.layers.len());

    let hidden = Tensor::zeros((1, 4, cfg.hidden_size), DType::F32, &device)?;
    decoder.forward(&hidden, None, None, Some(&mut cache), true)?;
    assert!(cache.seq_len().is_some());

    {
        let mut guard = decoder.prompt_guard(&mut cache);
        assert!(guard.cache().seq_len().is_some());
    }

    assert!(cache.seq_len().is_none());
    Ok(())
}
