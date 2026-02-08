use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

const DEFAULT_CONFIG_PATHS: &[&str] = &["GLM-OCR/config.json", "config.json"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmOcrRopeParameters {
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmOcrTextConfig {
    pub model_type: String,
    pub pad_token_id: i64,
    pub vocab_size: usize,
    pub eos_token_id: Vec<i64>,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_nextn_predict_layers: Option<usize>,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub dtype: Option<String>,
    pub rope_parameters: GlmOcrRopeParameters,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmOcrVisionConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub attention_bias: bool,
    #[serde(default = "default_vision_attention_dropout")]
    pub attention_dropout: f32,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub initializer_range: f32,
    pub image_size: usize,
    pub patch_size: usize,
    pub out_hidden_size: usize,
    pub rms_norm_eps: f64,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    #[serde(default = "default_vision_in_channels")]
    pub in_channels: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmOcrConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub text_config: GlmOcrTextConfig,
    pub vision_config: GlmOcrVisionConfig,
    pub image_start_token_id: i64,
    pub image_end_token_id: i64,
    pub video_start_token_id: i64,
    pub video_end_token_id: i64,
    pub image_token_id: i64,
    pub video_token_id: i64,
    #[serde(default)]
    pub transformers_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmPreprocessorConfig {
    pub size: GlmPreprocessorSize,
    pub do_rescale: bool,
    #[serde(default = "default_preprocessor_rescale_factor")]
    pub rescale_factor: f64,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    #[serde(rename = "merge_size")]
    pub spatial_merge_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlmPreprocessorSize {
    pub shortest_edge: usize,
    pub longest_edge: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GlmGenerationConfig {
    #[serde(default)]
    pub do_sample: Option<bool>,
    #[serde(default)]
    pub pad_token_id: Option<i64>,
    #[serde(default, deserialize_with = "deserialize_eos_ids")]
    pub eos_token_id: Vec<i64>,
}

pub struct LoadedGlmConfig {
    pub value: GlmOcrConfig,
    pub path: PathBuf,
}

pub struct LoadedGlmPreprocessor {
    pub value: GlmPreprocessorConfig,
}

pub fn load_config(path: Option<&Path>) -> Result<LoadedGlmConfig> {
    let resolved =
        resolve_config_path(path).ok_or_else(|| anyhow!("failed to locate GLM-OCR config file"))?;
    let raw = fs::read_to_string(&resolved)
        .with_context(|| format!("failed to read config at {}", resolved.display()))?;
    let value: GlmOcrConfig = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse config at {}", resolved.display()))?;
    Ok(LoadedGlmConfig {
        value,
        path: resolved,
    })
}

pub fn load_preprocessor(config_path: &Path) -> Result<LoadedGlmPreprocessor> {
    let parent = config_path
        .parent()
        .ok_or_else(|| anyhow!("config path {} has no parent", config_path.display()))?;
    let path = parent.join("preprocessor_config.json");
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("failed to read preprocessor config at {}", path.display()))?;
    let value: GlmPreprocessorConfig = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse preprocessor config at {}", path.display()))?;
    Ok(LoadedGlmPreprocessor { value })
}

pub fn load_generation_config(config_path: &Path) -> Result<Option<GlmGenerationConfig>> {
    let parent = config_path
        .parent()
        .ok_or_else(|| anyhow!("config path {} has no parent", config_path.display()))?;
    let path = parent.join("generation_config.json");
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("failed to read generation config at {}", path.display()))?;
    let value: GlmGenerationConfig = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse generation config at {}", path.display()))?;
    Ok(Some(value))
}

fn resolve_config_path(path: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = path {
        return Some(p.to_path_buf());
    }
    DEFAULT_CONFIG_PATHS
        .iter()
        .map(Path::new)
        .map(Path::to_path_buf)
        .find(|candidate| candidate.exists())
}

const fn default_preprocessor_rescale_factor() -> f64 {
    1.0 / 255.0
}

const fn default_vision_in_channels() -> usize {
    3
}

const fn default_vision_attention_dropout() -> f32 {
    0.0
}

fn deserialize_eos_ids<'de, D>(deserializer: D) -> std::result::Result<Vec<i64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Eos {
        Single(i64),
        Multi(Vec<i64>),
        None,
    }

    let value = Option::<Eos>::deserialize(deserializer)?;
    Ok(match value.unwrap_or(Eos::None) {
        Eos::Single(id) => vec![id],
        Eos::Multi(ids) => ids,
        Eos::None => Vec::new(),
    })
}
