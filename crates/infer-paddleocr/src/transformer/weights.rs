use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, quantized::QMatMul};
use candle_nn::VarBuilder;
use tracing::trace;

use crate::{
    config::PaddleOcrVlConfig,
    snapshot::{SnapshotLinear, SnapshotLinearMap},
};

#[derive(Debug)]
pub struct LinearWeights {
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub qmatmul: Option<Arc<QMatMul>>,
    pub out_dim: usize,
    pub in_dim: usize,
    pub label: String,
}

impl LinearWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: VarBuilder,
        out_dim: usize,
        in_dim: usize,
        use_bias: bool,
        mut snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let label = qualified_name(&vb, "weight");
        let device = vb.device();
        let mut weight: Option<Tensor> = None;
        let mut bias: Option<Tensor> = None;
        let mut qmatmul = None;
        if let Some(hit) = snapshot_hits
            .as_deref_mut()
            .and_then(|hits| hits.remove(&label))
        {
            match hit {
                SnapshotLinear::Quantized {
                    qmatmul: qm,
                    bias: snap_bias,
                } => {
                    trace!(
                        tensor = label,
                        backend = backend_label(device),
                        container = snapshot_label.unwrap_or("snapshot"),
                        source = "snapshot",
                        action = "quantized",
                        "paddle-linear"
                    );
                    qmatmul = Some(qm);
                    bias = snap_bias;
                }
                SnapshotLinear::Float {
                    weight: snap_weight,
                    bias: snap_bias,
                } => {
                    trace!(
                        tensor = label,
                        backend = backend_label(device),
                        container = snapshot_label.unwrap_or("snapshot"),
                        source = "snapshot",
                        action = "float",
                        "paddle-linear"
                    );
                    weight = Some(snap_weight);
                    bias = snap_bias;
                }
            }
        }
        if weight.is_none() && qmatmul.is_none() {
            weight = Some(
                vb.get((out_dim, in_dim), "weight")
                    .context("missing linear weight")?
                    .contiguous()?,
            );
        }
        if bias.is_none() && use_bias && vb.contains_tensor("bias") {
            bias = Some(
                vb.get(out_dim, "bias")
                    .context("missing linear bias")?
                    .contiguous()?,
            );
        }
        Ok(Self {
            weight,
            bias,
            qmatmul,
            out_dim,
            in_dim,
            label,
        })
    }

    pub fn matmul_2d(&self, input: &Tensor) -> Result<Tensor> {
        if let Some(qm) = &self.qmatmul {
            run_quantized_matmul(qm, input)
        } else {
            let weight = self
                .weight
                .as_ref()
                .context("linear weight missing for float matmul")?;
            let mut transposed = weight.transpose(0, 1)?;
            if transposed.dtype() != input.dtype() {
                transposed = transposed.to_dtype(input.dtype())?;
            }
            Ok(input.matmul(&transposed)?)
        }
    }
}

fn run_quantized_matmul(qm: &QMatMul, input: &Tensor) -> Result<Tensor> {
    let dtype = input.dtype();
    let device = input.device();
    let mut out = if device.is_cuda() || device.is_metal() {
        let activations = if dtype == DType::F32 {
            input.clone()
        } else {
            input.to_dtype(DType::F32)?
        };
        qm.forward(&activations)?
    } else {
        qm.forward(input)?
    };
    if out.dtype() != dtype {
        out = out.to_dtype(dtype)?;
    }
    Ok(out)
}

fn qualified_name(vb: &VarBuilder, tensor: &str) -> String {
    let prefix = vb.prefix();
    if prefix.is_empty() {
        tensor.to_string()
    } else {
        format!("{}.{}", prefix, tensor)
    }
}

fn backend_label(device: &Device) -> &'static str {
    if device.is_cuda() {
        "CUDA"
    } else if device.is_metal() {
        "Metal"
    } else {
        "CPU"
    }
}

#[derive(Debug)]
pub struct ErnieAttentionWeights {
    pub q_proj: LinearWeights,
    pub k_proj: LinearWeights,
    pub v_proj: LinearWeights,
    pub o_proj: LinearWeights,
}

impl ErnieAttentionWeights {
    pub fn load(
        vb: &VarBuilder,
        cfg: &PaddleOcrVlConfig,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.resolved_num_key_value_heads();
        let mut snapshot_hits = snapshot_hits;

        let attn_vb = vb.pp("self_attn");
        let q_proj = LinearWeights::load(
            attn_vb.pp("q_proj"),
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.use_bias,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        let k_proj = LinearWeights::load(
            attn_vb.pp("k_proj"),
            num_kv_heads * head_dim,
            cfg.hidden_size,
            cfg.use_bias,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        let v_proj = LinearWeights::load(
            attn_vb.pp("v_proj"),
            num_kv_heads * head_dim,
            cfg.hidden_size,
            cfg.use_bias,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        let o_proj = LinearWeights::load(
            attn_vb.pp("o_proj"),
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.use_bias,
            snapshot_hits.as_deref_mut(),
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

#[derive(Debug)]
pub struct ErnieMlpWeights {
    pub gate_proj: LinearWeights,
    pub up_proj: LinearWeights,
    pub down_proj: LinearWeights,
}

impl ErnieMlpWeights {
    pub fn load(
        vb: &VarBuilder,
        cfg: &PaddleOcrVlConfig,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let mlp_vb = vb.pp("mlp");
        let mut snapshot_hits = snapshot_hits;
        let gate_proj = LinearWeights::load(
            mlp_vb.pp("gate_proj"),
            cfg.intermediate_size,
            cfg.hidden_size,
            cfg.use_bias,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        let up_proj = LinearWeights::load(
            mlp_vb.pp("up_proj"),
            cfg.intermediate_size,
            cfg.hidden_size,
            cfg.use_bias,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        let down_proj = LinearWeights::load(
            mlp_vb.pp("down_proj"),
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.use_bias,
            snapshot_hits.as_deref_mut(),
            snapshot_label,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

#[derive(Debug)]
pub struct ErnieDecoderLayerWeights {
    pub attention: ErnieAttentionWeights,
    pub mlp: ErnieMlpWeights,
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
}

impl ErnieDecoderLayerWeights {
    pub fn load(
        vb: &VarBuilder,
        cfg: &PaddleOcrVlConfig,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let mut snapshot_hits = snapshot_hits;
        let attention =
            ErnieAttentionWeights::load(vb, cfg, snapshot_hits.as_deref_mut(), snapshot_label)?;
        let mlp = ErnieMlpWeights::load(vb, cfg, snapshot_hits.as_deref_mut(), snapshot_label)?;
        let input_layernorm = vb
            .pp("input_layernorm")
            .get(cfg.hidden_size, "weight")
            .context("missing input_layernorm.weight")?
            .contiguous()?;
        let post_attention_layernorm = vb
            .pp("post_attention_layernorm")
            .get(cfg.hidden_size, "weight")
            .context("missing post_attention_layernorm.weight")?
            .contiguous()?;
        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

#[derive(Debug)]
pub struct ErnieModelWeights {
    pub embed_tokens: Tensor,
    pub layers: Vec<ErnieDecoderLayerWeights>,
    pub final_norm: Tensor,
}

impl ErnieModelWeights {
    pub fn load(
        vb: &VarBuilder,
        cfg: &PaddleOcrVlConfig,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let embed_tokens = vb
            .pp("embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")
            .context("missing model.embed_tokens.weight")?
            .contiguous()?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut snapshot_hits = snapshot_hits;
        for idx in 0..cfg.num_hidden_layers {
            let layer_vb = vb.pp(format!("layers.{idx}"));
            layers.push(ErnieDecoderLayerWeights::load(
                &layer_vb,
                cfg,
                snapshot_hits.as_deref_mut(),
                snapshot_label,
            )?);
        }

        let final_norm = vb
            .pp("norm")
            .get(cfg.hidden_size, "weight")
            .context("missing model.norm.weight")?
            .contiguous()?;

        Ok(Self {
            embed_tokens,
            layers,
            final_norm,
        })
    }
}
