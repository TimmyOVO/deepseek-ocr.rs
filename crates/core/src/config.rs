use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    DeepseekV2,
    Qwen2,
    GlmOcr,
    PaddleOcrVl,
    DotsOcr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RopeConfig {
    Standard {
        theta: f64,
        rotary_dim: usize,
    },
    MultiModal {
        theta: f64,
        sections: Vec<usize>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrModelConfig {
    pub architecture: ModelArchitecture,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_head_dim: usize,
    pub v_head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope: RopeConfig,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
}

impl OcrModelConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.num_attention_heads > 0,
            "num_attention_heads must be > 0"
        );
        ensure!(
            self.hidden_size > 0 && self.hidden_size.is_multiple_of(self.num_attention_heads),
            "hidden_size {} must be divisible by num_attention_heads {}",
            self.hidden_size,
            self.num_attention_heads
        );
        ensure!(
            self.num_kv_heads > 0 && self.num_attention_heads.is_multiple_of(self.num_kv_heads),
            "num_attention_heads {} must be divisible by num_kv_heads {}",
            self.num_attention_heads,
            self.num_kv_heads
        );
        ensure!(self.head_dim > 0, "head_dim must be > 0");
        ensure!(self.kv_head_dim > 0, "kv_head_dim must be > 0");
        ensure!(self.v_head_dim > 0, "v_head_dim must be > 0");
        ensure!(
            self.max_position_embeddings > 0,
            "max_position_embeddings must be > 0"
        );
        ensure!(self.rms_norm_eps > 0.0, "rms_norm_eps must be > 0");
        match &self.rope {
            RopeConfig::Standard { theta, rotary_dim } => {
                ensure!(*theta > 0.0, "rope theta must be > 0");
                ensure!(*rotary_dim > 0, "rope rotary_dim must be > 0");
                ensure!(
                    *rotary_dim <= self.kv_head_dim,
                    "rope rotary_dim {} exceeds kv_head_dim {}",
                    rotary_dim,
                    self.kv_head_dim
                );
            }
            RopeConfig::MultiModal { theta, sections } => {
                ensure!(*theta > 0.0, "rope theta must be > 0");
                ensure!(!sections.is_empty(), "mrope sections must not be empty");
                let total = sections.iter().sum::<usize>();
                ensure!(
                    total == self.head_dim,
                    "mrope sections sum {} must equal head_dim {}",
                    total,
                    self.head_dim
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepseekRuntimeConfig {
    pub base: OcrModelConfig,
    pub moe_intermediate_size: Option<usize>,
    pub n_shared_experts: usize,
    pub n_routed_experts: usize,
    pub ep_size: usize,
    pub routed_scaling_factor: f32,
    pub kv_lora_rank: Option<usize>,
    pub q_lora_rank: Option<usize>,
    pub n_group: Option<usize>,
    pub topk_group: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub moe_layer_freq: usize,
    pub first_k_dense_replace: usize,
    pub norm_topk_prob: bool,
    pub topk_method: String,
    pub scoring_func: String,
    pub aux_loss_alpha: f32,
    pub seq_aux: bool,
    pub hidden_act: String,
    pub initializer_range: f32,
    pub pretraining_tp: usize,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub use_mla: bool,
    pub attn_implementation: Option<String>,
    pub rope_scaling: Option<Value>,
    pub torch_dtype: Option<String>,
    pub lm_head: Option<bool>,
    pub rm_head: Option<bool>,
    pub pad_token_id: Option<i64>,
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub extra: BTreeMap<String, Value>,
}

impl DeepseekRuntimeConfig {
    pub fn validate(&self) -> Result<()> {
        self.base.validate()?;
        ensure!(self.ep_size > 0, "ep_size must be > 0");
        ensure!(
            self.pretraining_tp > 0,
            "pretraining_tp must be > 0"
        );
        ensure!(
            self.moe_layer_freq > 0,
            "moe_layer_freq must be > 0"
        );
        ensure!(
            self.routed_scaling_factor > 0.0,
            "routed_scaling_factor must be > 0"
        );
        if self.n_routed_experts > 0 {
            ensure!(
                self.num_experts_per_tok.is_some(),
                "num_experts_per_tok is required when n_routed_experts > 0"
            );
        }
        if let Some(num_experts_per_tok) = self.num_experts_per_tok {
            ensure!(
                num_experts_per_tok > 0,
                "num_experts_per_tok must be > 0"
            );
            if self.n_routed_experts > 0 {
                ensure!(
                    num_experts_per_tok <= self.n_routed_experts,
                    "num_experts_per_tok {} must be <= n_routed_experts {}",
                    num_experts_per_tok,
                    self.n_routed_experts
                );
            }
        }
        Ok(())
    }
}
