use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::config::DotsOcrConfig;

pub const IMAGE_START_TOKEN: &str = "<|img|>";
pub const IMAGE_PAD_TOKEN: &str = "<|imgpad|>";
pub const IMAGE_END_TOKEN: &str = "<|endofimg|>";

#[derive(Debug, Clone)]
pub struct DotsImageTokens {
    pub start: u32,
    pub pad: u32,
    pub end: u32,
}

impl DotsImageTokens {
    pub fn resolve(tokenizer: &Tokenizer, cfg: &DotsOcrConfig) -> Result<Self> {
        let start = token_id(tokenizer, IMAGE_START_TOKEN)?;
        let pad = token_id(tokenizer, IMAGE_PAD_TOKEN)?;
        let end = token_id(tokenizer, IMAGE_END_TOKEN)?;
        ensure!(
            pad == cfg.image_token_id,
            "image_token_id mismatch: config={}, tokenizer={}",
            cfg.image_token_id,
            pad
        );
        Ok(Self { start, pad, end })
    }
}

#[derive(Debug, Clone, Default)]
pub struct DotsTokenizerConfig {
    pub chat_template: Option<String>,
}

pub fn load_tokenizer_config(path: Option<&Path>) -> Result<DotsTokenizerConfig> {
    let owned;
    let path = match path {
        Some(path) => path,
        None => {
            owned = default_tokenizer_config_path();
            &owned
        }
    };
    let bytes = std::fs::read(path).with_context(|| {
        format!(
            "failed to read dots.ocr tokenizer_config.json from {}",
            path.display()
        )
    })?;
    let value: Map<String, Value> = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "failed to parse dots.ocr tokenizer_config.json at {}",
            path.display()
        )
    })?;
    let chat_template = value
        .get("chat_template")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    Ok(DotsTokenizerConfig { chat_template })
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer.token_to_id(token).ok_or_else(|| {
        anyhow!(
            "tokenizer missing `{}` special token required for dots.ocr",
            token
        )
    })
}

fn default_tokenizer_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../dots.ocr/tokenizer_config.json")
}
