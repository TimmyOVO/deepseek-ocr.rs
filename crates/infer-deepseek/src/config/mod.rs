use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use deepseek_ocr_core::config::{
    DeepseekRuntimeConfig, ModelArchitecture, OcrModelConfig, RopeConfig,
};
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
        let mut merged: DeepseekV2Config = serde_json::from_value(primary_value)?;
        if let Some(language_cfg) = &self.language_config
            && let Some(freq) = language_cfg.model.moe_layer_freq_override
        {
            merged.moe_layer_freq = freq;
        }
        Ok(merged)
    }

    pub fn resolved_runtime_config(&self) -> Result<DeepseekRuntimeConfig> {
        let cfg = self.resolved_language_config()?;
        DeepseekRuntimeConfig::try_from(cfg)
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

impl TryFrom<DeepseekV2Config> for DeepseekRuntimeConfig {
    type Error = anyhow::Error;

    fn try_from(value: DeepseekV2Config) -> Result<Self, Self::Error> {
        ensure_non_zero(
            value.num_attention_heads,
            "num_attention_heads must be > 0",
        )?;
        ensure_non_zero(value.hidden_size, "hidden_size must be > 0")?;
        if !value.hidden_size.is_multiple_of(value.num_attention_heads) {
            bail!(
                "hidden_size {} must be divisible by num_attention_heads {}",
                value.hidden_size,
                value.num_attention_heads
            );
        }
        let head_dim = value.hidden_size / value.num_attention_heads;
        let num_kv_heads = resolve_or_default(value.num_key_value_heads, value.num_attention_heads);
        ensure_non_zero(num_kv_heads, "num_key_value_heads resolved to 0")?;
        if !value.num_attention_heads.is_multiple_of(num_kv_heads) {
            bail!(
                "num_attention_heads {} must be divisible by num_key_value_heads {}",
                value.num_attention_heads,
                num_kv_heads
            );
        }

        let kv_head_dim = resolve_or_default(value.qk_nope_head_dim, head_dim);
        let v_head_dim = resolve_or_default(value.v_head_dim, head_dim);
        let rotary_dim = resolve_or_default(value.qk_rope_head_dim, head_dim);

        let base = OcrModelConfig {
            architecture: ModelArchitecture::DeepseekV2,
            hidden_size: value.hidden_size,
            intermediate_size: value.intermediate_size,
            num_hidden_layers: value.num_hidden_layers,
            num_attention_heads: value.num_attention_heads,
            num_kv_heads,
            head_dim,
            kv_head_dim,
            v_head_dim,
            vocab_size: value.vocab_size,
            max_position_embeddings: value.max_position_embeddings,
            rms_norm_eps: f64::from(value.rms_norm_eps),
            rope: RopeConfig::Standard {
                theta: f64::from(value.rope_theta),
                rotary_dim,
            },
            use_cache: value.use_cache,
            tie_word_embeddings: value.tie_word_embeddings,
        };

        let runtime = DeepseekRuntimeConfig {
            base,
            moe_intermediate_size: value.moe_intermediate_size,
            n_shared_experts: value.n_shared_experts.unwrap_or(0),
            n_routed_experts: value.n_routed_experts.unwrap_or(0),
            ep_size: value.ep_size,
            routed_scaling_factor: value.routed_scaling_factor,
            kv_lora_rank: value.kv_lora_rank,
            q_lora_rank: value.q_lora_rank,
            n_group: value.n_group,
            topk_group: value.topk_group,
            num_experts_per_tok: value.num_experts_per_tok,
            moe_layer_freq: value.moe_layer_freq,
            first_k_dense_replace: value.first_k_dense_replace.unwrap_or(0),
            norm_topk_prob: value.norm_topk_prob,
            topk_method: value.topk_method.unwrap_or_else(|| "greedy".to_string()),
            scoring_func: value.scoring_func.unwrap_or_else(|| "softmax".to_string()),
            aux_loss_alpha: value.aux_loss_alpha,
            seq_aux: value.seq_aux,
            hidden_act: value.hidden_act,
            initializer_range: value.initializer_range,
            pretraining_tp: value.pretraining_tp,
            attention_bias: value.attention_bias,
            attention_dropout: value.attention_dropout,
            use_mla: value.use_mla,
            attn_implementation: value.attn_implementation,
            rope_scaling: value.rope_scaling,
            torch_dtype: value.torch_dtype,
            lm_head: value.lm_head,
            rm_head: value.rm_head,
            pad_token_id: value.pad_token_id,
            bos_token_id: value.bos_token_id,
            eos_token_id: value.eos_token_id,
            extra: value.extra,
        };
        runtime.validate()?;
        Ok(runtime)
    }
}

impl From<DeepseekRuntimeConfig> for DeepseekV2Config {
    fn from(value: DeepseekRuntimeConfig) -> Self {
        let (rope_theta, qk_rope_head_dim) = match value.base.rope {
            RopeConfig::Standard { theta, rotary_dim } => (theta as f32, Some(rotary_dim)),
            RopeConfig::MultiModal { theta, .. } => (theta as f32, Some(value.base.head_dim)),
        };
        Self {
            vocab_size: value.base.vocab_size,
            hidden_size: value.base.hidden_size,
            intermediate_size: value.base.intermediate_size,
            moe_intermediate_size: value.moe_intermediate_size,
            num_hidden_layers: value.base.num_hidden_layers,
            num_attention_heads: value.base.num_attention_heads,
            num_key_value_heads: Some(value.base.num_kv_heads),
            n_shared_experts: Some(value.n_shared_experts),
            n_routed_experts: Some(value.n_routed_experts),
            ep_size: value.ep_size,
            routed_scaling_factor: value.routed_scaling_factor,
            kv_lora_rank: value.kv_lora_rank,
            q_lora_rank: value.q_lora_rank,
            qk_rope_head_dim,
            v_head_dim: Some(value.base.v_head_dim),
            qk_nope_head_dim: Some(value.base.kv_head_dim),
            topk_method: Some(value.topk_method),
            n_group: value.n_group,
            topk_group: value.topk_group,
            num_experts_per_tok: value.num_experts_per_tok,
            moe_layer_freq: value.moe_layer_freq,
            moe_layer_freq_override: None,
            first_k_dense_replace: Some(value.first_k_dense_replace),
            norm_topk_prob: value.norm_topk_prob,
            scoring_func: Some(value.scoring_func),
            aux_loss_alpha: value.aux_loss_alpha,
            seq_aux: value.seq_aux,
            hidden_act: value.hidden_act,
            max_position_embeddings: value.base.max_position_embeddings,
            initializer_range: value.initializer_range,
            rms_norm_eps: value.base.rms_norm_eps as f32,
            use_cache: value.base.use_cache,
            pad_token_id: value.pad_token_id,
            bos_token_id: value.bos_token_id,
            eos_token_id: value.eos_token_id,
            pretraining_tp: value.pretraining_tp,
            tie_word_embeddings: value.base.tie_word_embeddings,
            attn_implementation: value.attn_implementation,
            rope_theta,
            rope_scaling: value.rope_scaling,
            attention_bias: value.attention_bias,
            attention_dropout: value.attention_dropout,
            use_mla: value.use_mla,
            torch_dtype: value.torch_dtype,
            lm_head: value.lm_head,
            rm_head: value.rm_head,
            extra: value.extra,
        }
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
    pub moe_layer_freq_override: Option<usize>,
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
            if let Value::Array(fallback_array) = fallback
                && target_array.is_empty()
            {
                *target = Value::Array(fallback_array.clone());
            }
        }
        Value::Null => {
            *target = fallback.clone();
        }
        _ => {}
    }
}

fn ensure_non_zero(value: usize, message: &str) -> Result<()> {
    if value == 0 {
        bail!("{message}");
    }
    Ok(())
}

fn resolve_or_default(value: Option<usize>, fallback: usize) -> usize {
    match value {
        Some(0) | None => fallback,
        Some(v) => v,
    }
}
