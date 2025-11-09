use std::fmt::Write as _;

use crate::config::DeepseekV2Config;
use anyhow::{Context, Result, ensure};
use candle_core::Tensor;
use candle_nn::VarBuilder;

/// Fully connected layer weights captured directly from safetensors via [`VarBuilder`].
#[derive(Debug, Clone)]
pub struct LinearWeights {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl LinearWeights {
    fn load(vb: &VarBuilder, out_dim: usize, in_dim: usize, bias: bool) -> Result<Self> {
        let weight = vb
            .get((out_dim, in_dim), "weight")
            .with_context(|| format!("missing linear weight `{}`", qualified_name(vb, "weight")))?;
        let bias =
            if bias && vb.contains_tensor("bias") {
                Some(vb.get(out_dim, "bias").with_context(|| {
                    format!("missing linear bias `{}`", qualified_name(vb, "bias"))
                })?)
            } else {
                None
            };
        Ok(Self { weight, bias })
    }
}

#[derive(Debug, Clone)]
pub struct RmsNormWeights {
    pub weight: Tensor,
}

impl RmsNormWeights {
    fn load(vb: &VarBuilder, hidden_size: usize) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight").with_context(|| {
            format!("missing rmsnorm weight `{}`", qualified_name(vb, "weight"))
        })?;
        Ok(Self { weight })
    }
}

#[derive(Debug, Clone)]
pub struct AttentionWeights {
    pub q_proj: LinearWeights,
    pub k_proj: LinearWeights,
    pub v_proj: LinearWeights,
    pub o_proj: LinearWeights,
}

impl AttentionWeights {
    fn load(cfg: &DeepseekV2Config, vb: &VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        ensure!(
            hidden_size % num_heads == 0,
            "hidden_size {hidden_size} not divisible by num_attention_heads {num_heads}"
        );
        let head_dim = hidden_size / num_heads;
        let num_kv_heads = cfg.num_key_value_heads.unwrap_or(num_heads);
        let kv_head_dim = head_dim;
        let v_head_dim = non_zero_or(cfg.v_head_dim, head_dim);
        let attn_vb = vb.pp("self_attn");

        let q_proj = LinearWeights::load(
            &attn_vb.pp("q_proj"),
            num_heads * head_dim,
            hidden_size,
            true,
        )?;
        let k_proj = LinearWeights::load(
            &attn_vb.pp("k_proj"),
            num_kv_heads * kv_head_dim,
            hidden_size,
            true,
        )?;
        let v_proj = LinearWeights::load(
            &attn_vb.pp("v_proj"),
            num_kv_heads * v_head_dim,
            hidden_size,
            true,
        )?;
        let o_proj = LinearWeights::load(
            &attn_vb.pp("o_proj"),
            hidden_size,
            num_heads * v_head_dim,
            true,
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DenseMlpWeights {
    pub gate_proj: LinearWeights,
    pub up_proj: LinearWeights,
    pub down_proj: LinearWeights,
}

impl DenseMlpWeights {
    fn load(vb: &VarBuilder, hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        let gate_proj =
            LinearWeights::load(&vb.pp("gate_proj"), intermediate_size, hidden_size, true)?;
        let up_proj = LinearWeights::load(&vb.pp("up_proj"), intermediate_size, hidden_size, true)?;
        let down_proj =
            LinearWeights::load(&vb.pp("down_proj"), hidden_size, intermediate_size, true)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

#[derive(Debug, Clone)]
pub struct MoeWeights {
    pub gate_weight: Tensor,
    pub experts: Vec<DenseMlpWeights>,
    pub shared_experts: Option<DenseMlpWeights>,
    pub aux_bias: Option<Tensor>,
}

impl MoeWeights {
    fn load(cfg: &DeepseekV2Config, layer_idx: usize, vb: &VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let moe_intermediate_size = cfg
            .moe_intermediate_size
            .with_context(|| "config.moe_intermediate_size missing for MoE layer")?;
        let num_routed = cfg
            .n_routed_experts
            .with_context(|| "config.n_routed_experts missing for MoE layer")?;
        ensure!(num_routed > 0, "n_routed_experts must be > 0 for MoE");

        let gate_weight = vb
            .pp("gate")
            .get((num_routed, hidden_size), "weight")
            .with_context(|| format!("missing MoE gate weight for layer {layer_idx}"))?;
        let aux_bias = if vb.pp("gate").contains_tensor("e_score_correction_bias") {
            Some(
                vb.pp("gate")
                    .get(num_routed, "e_score_correction_bias")
                    .with_context(|| {
                        format!("missing MoE gate e_score_correction_bias for layer {layer_idx}")
                    })?,
            )
        } else {
            None
        };

        let mut experts = Vec::with_capacity(num_routed);
        for expert_idx in 0..num_routed {
            let expert_vb = vb.pp(format!("experts.{expert_idx}"));
            let expert = DenseMlpWeights::load(&expert_vb, hidden_size, moe_intermediate_size)
                .with_context(|| {
                    format!("failed to load MoE expert {expert_idx} (layer {layer_idx})")
                })?;
            experts.push(expert);
        }

        let shared_experts = if let Some(count) = cfg.n_shared_experts.filter(|c| *c > 0) {
            let vb = vb.pp("shared_experts");
            let intermediate = moe_intermediate_size * count;
            Some(
                DenseMlpWeights::load(&vb, hidden_size, intermediate).with_context(|| {
                    format!("failed to load shared_experts for layer {layer_idx}")
                })?,
            )
        } else {
            None
        };

        Ok(Self {
            gate_weight,
            experts,
            shared_experts,
            aux_bias,
        })
    }
}

#[derive(Debug, Clone)]
pub enum MlpWeights {
    Dense(DenseMlpWeights),
    Moe(MoeWeights),
}

impl MlpWeights {
    fn load(cfg: &DeepseekV2Config, layer_idx: usize, vb: &VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        if should_use_moe(cfg, layer_idx) {
            MoeWeights::load(cfg, layer_idx, vb).map(MlpWeights::Moe)
        } else {
            DenseMlpWeights::load(vb, hidden_size, intermediate_size).map(MlpWeights::Dense)
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransformerBlockWeights {
    pub attention: AttentionWeights,
    pub mlp: MlpWeights,
    pub input_layernorm: RmsNormWeights,
    pub post_attention_layernorm: RmsNormWeights,
}

impl TransformerBlockWeights {
    pub fn load(cfg: &DeepseekV2Config, layer_idx: usize, vb: &VarBuilder) -> Result<Self> {
        let attention = AttentionWeights::load(cfg, vb)?;
        let mlp = MlpWeights::load(cfg, layer_idx, &vb.pp("mlp"))?;
        let input_layernorm = RmsNormWeights::load(&vb.pp("input_layernorm"), cfg.hidden_size)?;
        let post_attention_layernorm =
            RmsNormWeights::load(&vb.pp("post_attention_layernorm"), cfg.hidden_size)?;
        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TransformerWeights {
    pub layers: Vec<TransformerBlockWeights>,
}

impl TransformerWeights {
    pub fn load(cfg: &DeepseekV2Config, vb: &VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_vb = vb.pp(format!("layers.{layer_idx}"));
            let layer = TransformerBlockWeights::load(cfg, layer_idx, &layer_vb)
                .with_context(|| format!("failed to load transformer layer `{layer_idx}`"))?;
            layers.push(layer);
        }
        Ok(Self { layers })
    }
}

#[derive(Debug, Clone)]
pub struct DeepseekLanguageModelWeights {
    pub token_embedding: Tensor,
    pub transformer: TransformerWeights,
    pub final_layernorm: RmsNormWeights,
    pub lm_head: Tensor,
}

impl DeepseekLanguageModelWeights {
    pub fn load(cfg: &DeepseekV2Config, vb: &VarBuilder) -> Result<Self> {
        let model_vb = vb.pp("model");
        let token_embedding = model_vb
            .pp("embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")
            .with_context(|| {
                format!(
                    "missing token embedding `{}`",
                    qualified_name(&model_vb.pp("embed_tokens"), "weight")
                )
            })?;
        let token_embedding = token_embedding.contiguous()?;
        let transformer = TransformerWeights::load(cfg, &model_vb)?;
        let final_layernorm = RmsNormWeights::load(&model_vb.pp("norm"), cfg.hidden_size)
            .with_context(|| {
                format!(
                    "missing final layernorm `{}`",
                    qualified_name(&model_vb.pp("norm"), "weight")
                )
            })?;
        let lm_head_vb = vb.pp("lm_head");
        let lm_head = lm_head_vb
            .get((cfg.vocab_size, cfg.hidden_size), "weight")
            .with_context(|| {
                format!(
                    "missing lm_head weight `{}`",
                    qualified_name(&lm_head_vb, "weight")
                )
            })?;
        let lm_head = lm_head.contiguous()?;

        if cfg.tie_word_embeddings {
            ensure!(
                token_embedding.shape().dims() == lm_head.shape().dims(),
                "tie_word_embeddings enabled but embedding/logit weights differ: {:?} vs {:?}",
                token_embedding.shape().dims(),
                lm_head.shape().dims()
            );
        }

        Ok(Self {
            token_embedding,
            transformer,
            final_layernorm,
            lm_head,
        })
    }
}

fn should_use_moe(cfg: &DeepseekV2Config, layer_idx: usize) -> bool {
    let num_routed = cfg.n_routed_experts.unwrap_or(0);
    if num_routed == 0 {
        return false;
    }
    let first_dense = cfg.first_k_dense_replace.unwrap_or(0);
    if layer_idx < first_dense {
        return false;
    }
    layer_idx % cfg.moe_layer_freq == 0
}

fn non_zero_or(value: Option<usize>, fallback: usize) -> usize {
    match value {
        Some(v) if v > 0 => v,
        _ => fallback,
    }
}

fn qualified_name(vb: &VarBuilder, tensor: &str) -> String {
    let prefix = vb.prefix();
    if prefix.is_empty() {
        tensor.to_string()
    } else {
        let mut composed = String::with_capacity(prefix.len() + tensor.len() + 1);
        let _ = write!(composed, "{prefix}.{tensor}");
        composed
    }
}
