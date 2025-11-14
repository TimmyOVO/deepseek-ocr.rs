use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

const DEFAULT_CONFIG_PATHS: &[&str] = &["PaddleOCR-VL/config.json", "config.json"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaddleOcrVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    #[serde(default)]
    pub hidden_act: String,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_tokens_per_second")]
    pub tokens_per_second: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    #[serde(default, rename = "rope_type")]
    pub rope_type: Option<String>,
    #[serde(default, rename = "type")]
    pub ty: Option<String>,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaddleOcrVlConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub image_token_id: Option<i64>,
    #[serde(default)]
    pub video_token_id: Option<i64>,
    #[serde(default)]
    pub vision_start_token_id: Option<i64>,
    #[serde(default)]
    pub pad_token_id: Option<i64>,
    #[serde(default)]
    pub bos_token_id: Option<i64>,
    #[serde(default)]
    pub eos_token_id: Option<i64>,
    #[serde(default)]
    pub attention_probs_dropout_prob: f32,
    #[serde(default)]
    pub hidden_dropout_prob: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default)]
    pub use_cache: bool,
    #[serde(default)]
    pub use_flash_attention: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub hidden_act: String,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub weight_share_add_bias: bool,
    #[serde(default = "default_compression_ratio")]
    pub compression_ratio: f32,
    #[serde(default)]
    pub use_3d_rope: bool,
    #[serde(default)]
    pub rope_is_neox_style: bool,
    pub vision_config: PaddleOcrVisionConfig,
}

impl PaddleOcrVisionConfig {
    pub fn merge_kernel_area(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size
    }
}

impl PaddleOcrVlConfig {
    pub fn resolved_num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn dtype_hint(&self) -> Option<&str> {
        self.torch_dtype.as_deref()
    }
}

pub struct LoadedPaddleConfig {
    pub value: PaddleOcrVlConfig,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct GenerationOverrides {
    #[serde(default)]
    bos_token_id: Option<i64>,
    #[serde(default)]
    eos_token_id: Option<i64>,
    #[serde(default)]
    pad_token_id: Option<i64>,
}

pub fn load_config(path: Option<&Path>) -> Result<LoadedPaddleConfig> {
    let resolved = resolve_config_path(path)
        .ok_or_else(|| anyhow!("failed to locate PaddleOCR-VL config file"))?;
    let raw = fs::read_to_string(&resolved)
        .with_context(|| format!("failed to read config at {}", resolved.display()))?;
    let mut value: PaddleOcrVlConfig = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse config at {}", resolved.display()))?;
    if let Some(overrides) = load_generation_overrides(&resolved)? {
        apply_generation_overrides(&mut value, &overrides);
    }
    Ok(LoadedPaddleConfig {
        value,
        path: resolved,
    })
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

const fn default_layer_norm_eps() -> f32 {
    1e-6
}

const fn default_spatial_merge_size() -> usize {
    2
}

const fn default_temporal_patch_size() -> usize {
    1
}

const fn default_tokens_per_second() -> usize {
    2
}

const fn default_rms_norm_eps() -> f32 {
    1e-5
}

const fn default_rope_theta() -> f64 {
    10000.0
}

const fn default_compression_ratio() -> f32 {
    1.0
}

fn load_generation_overrides(config_path: &Path) -> Result<Option<GenerationOverrides>> {
    let Some(parent) = config_path.parent() else {
        return Ok(None);
    };
    let candidate = parent.join("generation_config.json");
    if !candidate.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&candidate).with_context(|| {
        format!(
            "failed to read generation config at {}",
            candidate.display()
        )
    })?;
    let overrides: GenerationOverrides = serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to parse generation config at {}",
            candidate.display()
        )
    })?;
    Ok(Some(overrides))
}

fn apply_generation_overrides(cfg: &mut PaddleOcrVlConfig, overrides: &GenerationOverrides) {
    if cfg.bos_token_id.is_none() {
        cfg.bos_token_id = overrides.bos_token_id;
    }
    if cfg.eos_token_id.is_none() {
        cfg.eos_token_id = overrides.eos_token_id;
    }
    if cfg.pad_token_id.is_none() {
        cfg.pad_token_id = overrides.pad_token_id;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_config_populates_eos_from_generation_config() -> Result<()> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate directory has parent")
            .parent()
            .expect("workspace root");
        let config_path = root.join("PaddleOCR-VL/config.json");
        let loaded = load_config(Some(&config_path))?;
        assert_eq!(loaded.value.eos_token_id, Some(2));
        Ok(())
    }
}
