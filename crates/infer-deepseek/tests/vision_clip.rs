mod common;

use anyhow::Result;
use candle_core::{DType, Tensor};
use common::test_utils::workspace_path;
use deepseek_ocr_infer_deepseek::{
    config::load_ocr_config,
    vision::clip::{ClipVisionParams, adapt_position_embedding_for_tests},
};

#[test]
fn clip_params_match_config() -> Result<()> {
    let cfg_path = workspace_path("DeepSeek-OCR/config.json");
    let cfg = load_ocr_config(Some(&cfg_path))?;
    let params = ClipVisionParams::from_config(&cfg)?;
    assert_eq!(params.hidden_size, 1024);
    assert_eq!(params.num_heads, 16);
    assert_eq!(params.num_layers, 24);
    assert_eq!(params.patch_size, 14);
    assert_eq!(params.image_size, 224);
    Ok(())
}

#[test]
fn adapt_position_embedding_downsamples_to_target() -> Result<()> {
    let device = candle_core::Device::Cpu;
    let table = Tensor::zeros((1, 257, 1024), DType::F32, &device)?.contiguous()?;
    let adapted = adapt_position_embedding_for_tests(&table, 101)?;
    let (batch, tokens, hidden) = adapted.shape().dims3()?;
    assert_eq!(batch, 1);
    assert_eq!(tokens, 101);
    assert_eq!(hidden, 1024);
    Ok(())
}
