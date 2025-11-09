use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;

static DEFAULT_CONFIG_PATHS: Lazy<[&str; 2]> =
    Lazy::new(|| ["DeepSeek-OCR/config.json", "config.json"]);

/// Load the top-level DeepSeek OCR configuration from disk.
pub fn load_ocr_config(path: Option<&Path>) -> Result<DeepseekOcrConfig> {
    let resolved = match path {
        Some(p) => p.to_path_buf(),
        None => resolve_default_config_path()
            .context("failed to locate DeepSeek OCR config file in default locations")?,
    };
    let data = fs::read_to_string(&resolved)
        .with_context(|| format!("failed to read config file {}", resolved.display()))?;
    let config = serde_json::from_str::<DeepseekOcrConfig>(&data)
        .with_context(|| format!("failed to parse config file {}", resolved.display()))?;
    Ok(config)
}

fn resolve_default_config_path() -> Option<PathBuf> {
    DEFAULT_CONFIG_PATHS
        .iter()
        .map(Path::new)
        .map(Path::to_path_buf)
        .find(|candidate| candidate.exists())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepseekOcrConfig {
    #[serde(rename = "_name_or_path")]
    pub name_or_path: Option<String>,
    #[serde(default)]
    pub candidate_resolutions: Vec<[u32; 2]>,
    #[serde(default)]
    pub global_view_pos: Option<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub auto_map: BTreeMap<String, String>,
    #[serde(default)]
    pub language_config: Option<LanguageConfig>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub projector_config: Option<ProjectorConfig>,
    #[serde(default)]
    pub tile_tag: Option<String>,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub transformers_version: Option<String>,
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
    #[serde(flatten)]
    pub language_defaults: Option<DeepseekV2Config>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl DeepseekOcrConfig {
    pub fn resolved_language_config(&self) -> Result<DeepseekV2Config> {
        let mut primary_value = if let Some(language_cfg) = &self.language_config {
            serde_json::to_value(&language_cfg.model)?
        } else if let Some(defaults) = &self.language_defaults {
            serde_json::to_value(defaults)?
        } else {
            bail!(
                "language configuration missing (neither language_config nor top-level defaults provided)"
            );
        };

        if let Some(defaults) = &self.language_defaults {
            let defaults_value = serde_json::to_value(defaults)?;
            merge_missing(&mut primary_value, &defaults_value);
        }
        let merged: DeepseekV2Config = serde_json::from_value(primary_value)?;
        Ok(merged)
    }

    pub fn language_torch_dtype(&self) -> Option<&str> {
        self.language_config
            .as_ref()
            .and_then(|cfg| cfg.torch_dtype.as_deref())
            .or_else(|| {
                self.language_defaults
                    .as_ref()
                    .and_then(|cfg| cfg.torch_dtype.as_deref())
            })
    }

    pub fn resolved_projector_config(&self) -> Result<ProjectorConfig> {
        self.projector_config
            .clone()
            .context("projector_config missing from DeepseekOcrConfig")
    }

    pub fn resolved_vision_backbone(&self, name: &str) -> Option<VisionBackboneConfig> {
        self.vision_config
            .as_ref()
            .and_then(|vision| vision.width.get(name))
            .cloned()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub auto_map: BTreeMap<String, String>,
    #[serde(flatten)]
    pub model: DeepseekV2Config,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepseekV2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub n_shared_experts: Option<usize>,
    #[serde(default)]
    pub n_routed_experts: Option<usize>,
    #[serde(default = "default_ep_size")]
    pub ep_size: usize,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub kv_lora_rank: Option<usize>,
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    #[serde(default)]
    pub qk_rope_head_dim: Option<usize>,
    #[serde(default)]
    pub v_head_dim: Option<usize>,
    #[serde(default)]
    pub qk_nope_head_dim: Option<usize>,
    #[serde(default)]
    pub topk_method: Option<String>,
    #[serde(default)]
    pub n_group: Option<usize>,
    #[serde(default)]
    pub topk_group: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: usize,
    #[serde(default)]
    pub first_k_dense_replace: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub scoring_func: Option<String>,
    #[serde(default = "default_aux_loss_alpha")]
    pub aux_loss_alpha: f32,
    #[serde(default = "default_seq_aux")]
    pub seq_aux: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    #[serde(default)]
    pub pad_token_id: Option<i64>,
    #[serde(default)]
    pub bos_token_id: Option<i64>,
    #[serde(default)]
    pub eos_token_id: Option<i64>,
    #[serde(default = "default_pretraining_tp")]
    pub pretraining_tp: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default, rename = "_attn_implementation")]
    pub attn_implementation: Option<String>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_scaling: Option<Value>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default = "default_use_mla")]
    pub use_mla: bool,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub lm_head: Option<bool>,
    #[serde(default)]
    pub rm_head: Option<bool>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectorConfig {
    #[serde(default)]
    pub input_dim: Option<usize>,
    #[serde(default)]
    pub model_type: Option<String>,
    pub n_embed: usize,
    pub projector_type: String,
    #[serde(default)]
    pub depth: Option<usize>,
    #[serde(default)]
    pub mlp_ratio: Option<f32>,
    #[serde(default)]
    pub token_pooling: Option<bool>,
    #[serde(default)]
    pub downsample_ratio: Option<usize>,
    #[serde(default)]
    pub channel_div: Option<f32>,
    #[serde(default)]
    pub conv_fusion_high_low_features: Option<bool>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    #[serde(default)]
    pub image_size: Option<usize>,
    #[serde(default)]
    pub mlp_ratio: Option<f32>,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub width: BTreeMap<String, VisionBackboneConfig>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VisionBackboneConfig {
    #[serde(default)]
    pub heads: Option<usize>,
    #[serde(default)]
    pub image_size: Option<usize>,
    #[serde(default)]
    pub layers: Option<usize>,
    #[serde(default)]
    pub patch_size: Option<usize>,
    #[serde(default)]
    pub width: Option<usize>,
    #[serde(default)]
    pub downsample_channels: Option<Vec<usize>>,
    #[serde(default)]
    pub global_attn_indexes: Option<Vec<usize>>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

fn default_ep_size() -> usize {
    1
}

fn default_routed_scaling_factor() -> f32 {
    1.0
}

fn default_moe_layer_freq() -> usize {
    1
}

fn default_aux_loss_alpha() -> f32 {
    0.001
}

fn default_seq_aux() -> bool {
    true
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_initializer_range() -> f32 {
    0.02
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_use_cache() -> bool {
    true
}

fn default_pretraining_tp() -> usize {
    1
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_use_mla() -> bool {
    true
}

fn merge_missing(target: &mut Value, fallback: &Value) {
    match target {
        Value::Object(target_map) => {
            if let Value::Object(fallback_map) = fallback {
                for (key, fallback_value) in fallback_map {
                    match target_map.get_mut(key) {
                        Some(target_value) => {
                            if target_value.is_null() {
                                target_map.insert(key.clone(), fallback_value.clone());
                            } else {
                                merge_missing(target_value, fallback_value);
                            }
                        }
                        None => {
                            target_map.insert(key.clone(), fallback_value.clone());
                        }
                    }
                }
            }
        }
        Value::Array(target_array) => {
            if let Value::Array(fallback_array) = fallback {
                if target_array.is_empty() {
                    *target = Value::Array(fallback_array.clone());
                }
            }
        }
        Value::Null => {
            *target = fallback.clone();
        }
        _ => {}
    }
}
