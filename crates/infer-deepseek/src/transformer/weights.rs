use std::{
    fmt::{self, Write as _},
    sync::Arc,
};

use crate::{
    config::DeepseekV2Config,
    quant_snapshot::{
        LinearSpec, QuantizedSnapshot, SnapshotLinear, SnapshotLinearMap, SnapshotLoadPlan,
    },
    quantization::{
        LinearLayerGroup, QuantModule, QuantizationOutcome, QuantizationState, backend_label,
    },
};
use anyhow::{Context, Result, ensure};
use candle_core::{Tensor, quantized::QMatMul};
use candle_nn::VarBuilder;
use tracing::trace;

/// Fully connected layer weights captured directly from safetensors via [`VarBuilder`].
#[derive(Clone)]
pub struct LinearWeights {
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub qmatmul: Option<Arc<QMatMul>>,
    pub out_dim: usize,
    pub in_dim: usize,
    pub label: String,
}

impl fmt::Debug for LinearWeights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinearWeights")
            .field("has_weight", &self.weight.is_some())
            .field("bias", &self.bias)
            .field("qmatmul", &self.qmatmul.is_some())
            .field("out_dim", &self.out_dim)
            .field("in_dim", &self.in_dim)
            .field("label", &self.label)
            .finish()
    }
}

impl LinearWeights {
    fn snapshot_spec(vb: &VarBuilder, out_dim: usize, in_dim: usize) -> LinearSpec {
        LinearSpec::new(qualified_name(vb, "weight"), out_dim, in_dim)
    }

    fn load(
        vb: &VarBuilder,
        out_dim: usize,
        in_dim: usize,
        bias: bool,
        group: LinearLayerGroup,
        module: QuantModule,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let label = qualified_name(vb, "weight");
        let mut weight = Some(
            vb.get((out_dim, in_dim), "weight")
                .with_context(|| format!("missing linear weight `{label}`"))?
                .contiguous()?,
        );
        let mut bias_tensor: Option<Tensor> = None;
        let device = vb.device();
        let quant = QuantizationState::global();
        let mut qmatmul = None;
        // If snapshot hits were preloaded, prefer them regardless of env quant targets/kind.
        if let Some(hits) = snapshot_hits {
            if let Some(hit) = hits.remove(&label) {
                let container = snapshot_label.unwrap_or("snapshot");
                match hit {
                    SnapshotLinear::Quantized { qmatmul: qm, bias } => {
                        let path = if device.is_cuda() || device.is_metal() {
                            "kernel_upcast"
                        } else {
                            "kernel"
                        };
                        trace!(
                            tensor = label,
                            ?group,
                            in_dim,
                            out_dim = out_dim,
                            backend = backend_label(device),
                            path,
                            container,
                            source = "snapshot",
                            action = "quantized",
                            "quant-linear"
                        );
                        quant.record_attempt(module, QuantizationOutcome::Quantized);
                        bias_tensor = bias;
                        qmatmul = Some(qm);
                        weight = None;
                    }
                    SnapshotLinear::Float {
                        weight: snapshot_weight,
                        bias,
                    } => {
                        trace!(
                            tensor = label,
                            ?group,
                            in_dim,
                            out_dim = out_dim,
                            backend = backend_label(device),
                            path = "snapshot-float",
                            container,
                            source = "snapshot",
                            action = "float",
                            "quant-linear"
                        );
                        quant.record_attempt(module, QuantizationOutcome::Fallback);
                        bias_tensor = bias;
                        weight = Some(snapshot_weight);
                    }
                }
            }
        }
        // No runtime quantization fallback: use snapshot when available, otherwise float weights.
        if bias && bias_tensor.is_none() && vb.contains_tensor("bias") {
            bias_tensor = Some(
                vb.get(out_dim, "bias")
                    .with_context(|| {
                        format!("missing linear bias `{}`", qualified_name(vb, "bias"))
                    })?
                    .contiguous()?,
            );
        }
        Ok(Self {
            weight,
            bias: bias_tensor,
            qmatmul,
            out_dim,
            in_dim,
            label,
        })
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
    fn load(
        cfg: &DeepseekV2Config,
        vb: &VarBuilder,
        snapshot: Option<&QuantizedSnapshot>,
    ) -> Result<Self> {
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
        let q_vb = attn_vb.pp("q_proj");
        let k_vb = attn_vb.pp("k_proj");
        let v_vb = attn_vb.pp("v_proj");
        let o_vb = attn_vb.pp("o_proj");
        let mut plan = SnapshotLoadPlan::default();
        plan.push(LinearWeights::snapshot_spec(
            &q_vb,
            num_heads * head_dim,
            hidden_size,
        ));
        plan.push(LinearWeights::snapshot_spec(
            &k_vb,
            num_kv_heads * kv_head_dim,
            hidden_size,
        ));
        plan.push(LinearWeights::snapshot_spec(
            &v_vb,
            num_kv_heads * v_head_dim,
            hidden_size,
        ));
        plan.push(LinearWeights::snapshot_spec(
            &o_vb,
            hidden_size,
            num_heads * v_head_dim,
        ));
        let mut snapshot_hits = plan.execute(snapshot, vb.device(), None)?;
        let snapshot_label = snapshot.map(|s| s.container_label());

        let q_proj = LinearWeights::load(
            &q_vb,
            num_heads * head_dim,
            hidden_size,
            true,
            LinearLayerGroup::Text,
            QuantModule::TextLinear,
            snapshot_hits.as_mut().map(|hits| hits),
            snapshot_label,
        )?;
        let k_proj = LinearWeights::load(
            &k_vb,
            num_kv_heads * kv_head_dim,
            hidden_size,
            true,
            LinearLayerGroup::Text,
            QuantModule::TextLinear,
            snapshot_hits.as_mut().map(|hits| hits),
            snapshot_label,
        )?;
        let v_proj = LinearWeights::load(
            &v_vb,
            num_kv_heads * v_head_dim,
            hidden_size,
            true,
            LinearLayerGroup::Text,
            QuantModule::TextLinear,
            snapshot_hits.as_mut().map(|hits| hits),
            snapshot_label,
        )?;
        let o_proj = LinearWeights::load(
            &o_vb,
            hidden_size,
            num_heads * v_head_dim,
            true,
            LinearLayerGroup::Text,
            QuantModule::TextLinear,
            snapshot_hits.as_mut().map(|hits| hits),
            snapshot_label,
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
    fn load(
        vb: &VarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        snapshot: Option<&QuantizedSnapshot>,
    ) -> Result<Self> {
        let gate_vb = vb.pp("gate_proj");
        let up_vb = vb.pp("up_proj");
        let down_vb = vb.pp("down_proj");
        let mut plan = SnapshotLoadPlan::default();
        plan.push(LinearWeights::snapshot_spec(
            &gate_vb,
            intermediate_size,
            hidden_size,
        ));
        plan.push(LinearWeights::snapshot_spec(
            &up_vb,
            intermediate_size,
            hidden_size,
        ));
        plan.push(LinearWeights::snapshot_spec(
            &down_vb,
            hidden_size,
            intermediate_size,
        ));
        let mut snapshot_hits = plan.execute(snapshot, vb.device(), None)?;
        let snapshot_label = snapshot.map(|s| s.container_label());

        let gate_proj = LinearWeights::load(
            &gate_vb,
            intermediate_size,
            hidden_size,
            true,
            LinearLayerGroup::Text,
            QuantModule::TextLinear,
            snapshot_hits.as_mut().map(|hits| hits),
            snapshot_label,
        )?;
        let up_proj = LinearWeights::load(
            &up_vb,
            intermediate_size,
            hidden_size,
            true,
            LinearLayerGroup::Text,
            QuantModule::TextLinear,
            snapshot_hits.as_mut().map(|hits| hits),
            snapshot_label,
        )?;
        let down_proj = LinearWeights::load(
            &down_vb,
            hidden_size,
            intermediate_size,
            true,
            LinearLayerGroup::Text,
            QuantModule::TextLinear,
            snapshot_hits.as_mut().map(|hits| hits),
            snapshot_label,
        )?;
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
    fn load(
        cfg: &DeepseekV2Config,
        layer_idx: usize,
        vb: &VarBuilder,
        snapshot: Option<&QuantizedSnapshot>,
    ) -> Result<Self> {
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
            let expert =
                DenseMlpWeights::load(&expert_vb, hidden_size, moe_intermediate_size, snapshot)
                    .with_context(|| {
                        format!("failed to load MoE expert {expert_idx} (layer {layer_idx})")
                    })?;
            experts.push(expert);
        }

        let shared_experts = if let Some(count) = cfg.n_shared_experts.filter(|c| *c > 0) {
            let vb = vb.pp("shared_experts");
            let intermediate = moe_intermediate_size * count;
            Some(
                DenseMlpWeights::load(&vb, hidden_size, intermediate, snapshot).with_context(
                    || format!("failed to load shared_experts for layer {layer_idx}"),
                )?,
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
    fn load(
        cfg: &DeepseekV2Config,
        layer_idx: usize,
        vb: &VarBuilder,
        snapshot: Option<&QuantizedSnapshot>,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        if should_use_moe(cfg, layer_idx) {
            MoeWeights::load(cfg, layer_idx, vb, snapshot).map(MlpWeights::Moe)
        } else {
            DenseMlpWeights::load(vb, hidden_size, intermediate_size, snapshot)
                .map(MlpWeights::Dense)
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
    pub fn load(
        cfg: &DeepseekV2Config,
        layer_idx: usize,
        vb: &VarBuilder,
        snapshot: Option<&QuantizedSnapshot>,
    ) -> Result<Self> {
        let attention = AttentionWeights::load(cfg, vb, snapshot)?;
        let mlp = MlpWeights::load(cfg, layer_idx, &vb.pp("mlp"), snapshot)?;
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
    pub fn load(
        cfg: &DeepseekV2Config,
        vb: &VarBuilder,
        snapshot: Option<&QuantizedSnapshot>,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_vb = vb.pp(format!("layers.{layer_idx}"));
            let layer = TransformerBlockWeights::load(cfg, layer_idx, &layer_vb, snapshot)
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
    pub lm_head_weight: Option<Tensor>,
    pub lm_head_q: Option<Arc<QMatMul>>,
    pub lm_out_dim: usize,
    pub lm_in_dim: usize,
    pub lm_head_label: String,
}

impl DeepseekLanguageModelWeights {
    pub fn load(
        cfg: &DeepseekV2Config,
        vb: &VarBuilder,
        snapshot: Option<&QuantizedSnapshot>,
    ) -> Result<Self> {
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
        let transformer = TransformerWeights::load(cfg, &model_vb, snapshot)?;
        let final_layernorm = RmsNormWeights::load(&model_vb.pp("norm"), cfg.hidden_size)
            .with_context(|| {
                format!(
                    "missing final layernorm `{}`",
                    qualified_name(&model_vb.pp("norm"), "weight")
                )
            })?;
        let lm_head_vb = vb.pp("lm_head");
        let lm_head_label = qualified_name(&lm_head_vb, "weight");
        let mut lm_head_weight = Some(
            lm_head_vb
                .get((cfg.vocab_size, cfg.hidden_size), "weight")
                .with_context(|| format!("missing lm_head weight `{}`", lm_head_label))?
                .contiguous()?,
        );

        if cfg.tie_word_embeddings {
            ensure!(
                token_embedding.shape().dims() == [cfg.vocab_size, cfg.hidden_size],
                "tie_word_embeddings enabled but embedding/logit weights differ: {:?} vs {:?}",
                token_embedding.shape().dims(),
                [cfg.vocab_size, cfg.hidden_size]
            );
        }

        // Try offline snapshot first, falling back to runtime quantization.
        let quant = QuantizationState::global();
        let mut lm_q = None;
        if let Some(snapshot) = snapshot {
            let mut plan = SnapshotLoadPlan::default();
            plan.push(LinearSpec::new(
                lm_head_label.clone(),
                cfg.vocab_size,
                cfg.hidden_size,
            ));
            let mut hits = plan.execute(Some(snapshot), vb.device(), None)?;
            if let Some(hit) = hits.as_mut().and_then(|map| map.remove(&lm_head_label)) {
                match hit {
                    SnapshotLinear::Quantized { qmatmul, bias: _ } => {
                        let path = if vb.device().is_cuda() || vb.device().is_metal() {
                            "kernel_upcast"
                        } else {
                            "kernel"
                        };
                        trace!(
                            tensor = lm_head_label,
                            module = "lm_head",
                            in_dim = cfg.hidden_size,
                            out_dim = cfg.vocab_size,
                            backend = backend_label(vb.device()),
                            path,
                            container = snapshot.container_label(),
                            source = "snapshot",
                            action = "quantized",
                            "quant-linear"
                        );
                        quant.record_attempt(QuantModule::LmHead, QuantizationOutcome::Quantized);
                        lm_q = Some(qmatmul);
                        lm_head_weight = None;
                    }
                    SnapshotLinear::Float { weight, bias: _ } => {
                        trace!(
                            tensor = lm_head_label,
                            module = "lm_head",
                            in_dim = cfg.hidden_size,
                            out_dim = cfg.vocab_size,
                            backend = backend_label(vb.device()),
                            path = "snapshot-float",
                            container = snapshot.container_label(),
                            source = "snapshot",
                            action = "float",
                            "quant-linear"
                        );
                        quant.record_attempt(QuantModule::LmHead, QuantizationOutcome::Fallback);
                        lm_head_weight = Some(weight);
                    }
                }
            }
        }

        Ok(Self {
            token_embedding,
            transformer,
            final_layernorm,
            lm_head_weight,
            lm_head_q: lm_q,
            lm_out_dim: cfg.vocab_size,
            lm_in_dim: cfg.hidden_size,
            lm_head_label,
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

pub(crate) fn qualified_name(vb: &VarBuilder, tensor: &str) -> String {
    let prefix = vb.prefix();
    if prefix.is_empty() {
        tensor.to_string()
    } else {
        let mut composed = String::with_capacity(prefix.len() + tensor.len() + 1);
        let _ = write!(composed, "{prefix}.{tensor}");
        composed
    }
}

// Runtime quantization path removed: no `maybe_quantize_linear` fallback.
