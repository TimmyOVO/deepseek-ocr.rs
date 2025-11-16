use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct DotsVisionConfig {
    pub embed_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub rms_norm_eps: f64,
    pub use_bias: bool,
    pub attn_implementation: Option<String>,
    pub initializer_range: f64,
    pub init_merger_std: f64,
    pub is_causal: bool,
    pub post_norm: bool,
}

#[derive(Debug, Clone)]
pub struct DotsOcrTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub max_window_layers: usize,
    pub sliding_window: Option<usize>,
    pub use_sliding_window: bool,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub initializer_range: f64,
    pub hidden_act: String,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(Debug, Clone)]
pub struct DotsOcrConfig {
    pub model_type: String,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub text: DotsOcrTextConfig,
    pub vision: DotsVisionConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct RawDotsOcrConfig {
    model_type: String,
    attention_bias: Option<bool>,
    attention_dropout: Option<f64>,
    hidden_act: Option<String>,
    hidden_size: usize,
    initializer_range: Option<f64>,
    intermediate_size: usize,
    max_position_embeddings: usize,
    max_window_layers: Option<usize>,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    sliding_window: Option<usize>,
    tie_word_embeddings: Option<bool>,
    use_cache: Option<bool>,
    use_sliding_window: Option<bool>,
    vocab_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    vision_config: DotsVisionConfig,
}
impl From<RawDotsOcrConfig> for DotsOcrConfig {
    fn from(raw: RawDotsOcrConfig) -> Self {
        let text = DotsOcrTextConfig {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: raw.rope_theta,
            max_position_embeddings: raw.max_position_embeddings,
            max_window_layers: raw.max_window_layers.unwrap_or(raw.num_hidden_layers),
            sliding_window: raw.sliding_window,
            use_sliding_window: raw.use_sliding_window.unwrap_or(false),
            attention_bias: raw.attention_bias.unwrap_or(true),
            attention_dropout: raw.attention_dropout.unwrap_or(0.0),
            initializer_range: raw.initializer_range.unwrap_or(0.02),
            hidden_act: raw.hidden_act.unwrap_or_else(|| "silu".to_string()),
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
            use_cache: raw.use_cache.unwrap_or(true),
            vocab_size: raw.vocab_size,
        };

        Self {
            model_type: raw.model_type,
            image_token_id: raw.image_token_id,
            video_token_id: raw.video_token_id,
            text,
            vision: raw.vision_config,
        }
    }
}

pub fn load_dots_config(path: Option<&Path>) -> Result<DotsOcrConfig> {
    let owned;
    let path = match path {
        Some(path) => path,
        None => {
            owned = default_config_path();
            &owned
        }
    };
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read dots.ocr config from {}", path.display()))?;
    let raw: RawDotsOcrConfig = serde_json::from_slice(&bytes)
        .with_context(|| format!("dots.ocr config `{}` contains invalid JSON", path.display()))?;
    Ok(raw.into())
}

fn default_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../dots.ocr/config.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_default_config_from_repo() {
        let cfg = load_dots_config(None).expect("config should be readable");
        assert_eq!(cfg.model_type, "dots_ocr");
        assert_eq!(cfg.image_token_id, 151665);
        assert_eq!(cfg.video_token_id, 151656);
        assert_eq!(cfg.text.num_hidden_layers, 28);
        assert_eq!(cfg.vision.num_hidden_layers, 42);
    }
}
