mod common;

use anyhow::{Context, Result};
use common::test_utils::workspace_path;
use deepseek_ocr_infer_deepseek::config::{DeepseekOcrConfig, load_ocr_config};

fn load_test_config() -> Result<DeepseekOcrConfig> {
    let path = workspace_path("DeepSeek-OCR/config.json");
    load_ocr_config(Some(&path))
}

#[test]
fn load_default_config() -> Result<()> {
    let config = load_test_config()?;
    assert_eq!(
        config
            .name_or_path
            .as_deref()
            .unwrap_or("deepseek-ai/DeepSeek-OCR"),
        "deepseek-ai/DeepSeek-OCR"
    );
    assert!(config.projector_config.is_some());
    assert!(config.vision_config.is_some());
    assert!(config.language_config.is_some() || config.language_defaults.is_some());
    Ok(())
}

#[test]
fn language_resolution_prefers_nested_config() -> Result<()> {
    let config = load_test_config()?;
    let language = config.resolved_language_config()?;
    assert_eq!(language.hidden_size, 1280);
    assert_eq!(language.num_hidden_layers, 12);
    assert_eq!(language.num_attention_heads, 10);
    assert_eq!(config.language_torch_dtype(), Some("bfloat16"));
    Ok(())
}

#[test]
fn projector_resolution_mirrors_config() -> Result<()> {
    let config = load_test_config()?;
    let projector = config.resolved_projector_config()?;
    assert_eq!(projector.projector_type, "linear");
    assert_eq!(projector.n_embed, 1280);
    Ok(())
}

#[test]
fn vision_backbone_lookup() -> Result<()> {
    let config = load_test_config()?;
    let sam = config
        .resolved_vision_backbone("sam_vit_b")
        .context("sam_vit_b config missing")?;
    assert_eq!(sam.width, Some(768));
    assert_eq!(sam.layers, Some(12));
    assert_eq!(sam.heads, Some(12));
    Ok(())
}
