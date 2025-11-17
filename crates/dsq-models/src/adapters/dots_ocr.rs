use anyhow::{Context, Result};
use deepseek_ocr_dsq::DsqTensorDType;
use serde_json::{Map, Value};

use crate::{
    adapters::helpers::{
        get_optional_nonzero, get_optional_usize, get_required_usize, root_object,
    },
    AdapterScope, LinearSpec, ModelAdapter, QuantContext,
};

pub struct DotsOcrAdapter;

impl ModelAdapter for DotsOcrAdapter {
    fn id(&self) -> &'static str {
        "dots-ocr"
    }

    fn supports(&self, cfg: &Value) -> bool {
        cfg.as_object()
            .and_then(|root| root.get("model_type"))
            .and_then(Value::as_str)
            .map(|ty| {
                ty.eq_ignore_ascii_case("dots_ocr") || ty.to_ascii_lowercase().contains("dots")
            })
            .unwrap_or(false)
    }

    fn discover(&self, cfg: &Value, scope: AdapterScope) -> Result<Vec<LinearSpec>> {
        let root = root_object(cfg)?;
        let text_cfg = dots_text_config(root).context("missing text configuration for DotsOCR")?;
        let mut specs = text_decoder_specs(text_cfg)?;
        if scope.includes_projector() {
            let vision_cfg =
                dots_vision_config(root).context("missing vision configuration for DotsOCR")?;
            specs.extend(vision_encoder_specs(vision_cfg)?);
            specs.extend(merger_specs(vision_cfg));
        }
        Ok(specs)
    }

    fn recommend_dtype(
        &self,
        tensor: &str,
        _in_dim: usize,
        ctx: &QuantContext,
    ) -> Option<DsqTensorDType> {
        // Prefer a slightly higher-precision dtype for the final projection when
        // the primary dtype permits it, mirroring DeepSeek/Paddle behaviour.
        if ctx.primary == DsqTensorDType::Q8_0 {
            return None;
        }
        if tensor == "lm_head.weight" {
            return Some(DsqTensorDType::Q8_0);
        }
        None
    }
}

fn dots_text_config(root: &Map<String, Value>) -> Option<&Map<String, Value>> {
    if root.contains_key("hidden_size") && root.contains_key("num_hidden_layers") {
        Some(root)
    } else {
        root.get("text").and_then(Value::as_object)
    }
}

fn dots_vision_config(root: &Map<String, Value>) -> Option<&Map<String, Value>> {
    if let Some(v) = root.get("vision_config").and_then(Value::as_object) {
        Some(v)
    } else {
        root.get("vision").and_then(Value::as_object)
    }
}

fn text_decoder_specs(root: &Map<String, Value>) -> Result<Vec<LinearSpec>> {
    let hidden_size = get_required_usize(root, "hidden_size")?;
    let intermediate = get_required_usize(root, "intermediate_size")?;
    let num_layers = get_required_usize(root, "num_hidden_layers")?;
    let num_heads = get_required_usize(root, "num_attention_heads")?;
    let num_kv_heads = get_optional_usize(root, "num_key_value_heads").unwrap_or(num_heads);
    let head_dim = hidden_size
        .checked_div(num_heads)
        .context("hidden_size must be divisible by num_attention_heads")?;
    let kv_dim = num_kv_heads.max(1) * head_dim;
    let vocab_size = get_required_usize(root, "vocab_size")?;
    let attention_bias = root
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    let mut specs = Vec::new();
    for layer_idx in 0..num_layers {
        let layer_prefix = format!("model.layers.{layer_idx}");
        let attn_prefix = format!("{layer_prefix}.self_attn");
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.q_proj.weight"),
            out_dim: hidden_size,
            in_dim: hidden_size,
            bias: bias_name(attention_bias, &attn_prefix, "q_proj.bias"),
        });
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.k_proj.weight"),
            out_dim: kv_dim,
            in_dim: hidden_size,
            bias: bias_name(attention_bias, &attn_prefix, "k_proj.bias"),
        });
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.v_proj.weight"),
            out_dim: kv_dim,
            in_dim: hidden_size,
            bias: bias_name(attention_bias, &attn_prefix, "v_proj.bias"),
        });
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.o_proj.weight"),
            out_dim: hidden_size,
            in_dim: hidden_size,
            bias: bias_name(attention_bias, &attn_prefix, "o_proj.bias"),
        });

        let mlp_prefix = format!("{layer_prefix}.mlp");
        specs.push(LinearSpec {
            name: format!("{mlp_prefix}.gate_proj.weight"),
            out_dim: intermediate,
            in_dim: hidden_size,
            bias: Some(format!("{mlp_prefix}.gate_proj.bias")),
        });
        specs.push(LinearSpec {
            name: format!("{mlp_prefix}.up_proj.weight"),
            out_dim: intermediate,
            in_dim: hidden_size,
            bias: Some(format!("{mlp_prefix}.up_proj.bias")),
        });
        specs.push(LinearSpec {
            name: format!("{mlp_prefix}.down_proj.weight"),
            out_dim: hidden_size,
            in_dim: intermediate,
            bias: Some(format!("{mlp_prefix}.down_proj.bias")),
        });
    }

    specs.push(LinearSpec {
        name: "lm_head.weight".to_string(),
        out_dim: vocab_size,
        in_dim: hidden_size,
        bias: None,
    });

    Ok(specs)
}

fn vision_encoder_specs(cfg: &Map<String, Value>) -> Result<Vec<LinearSpec>> {
    let embed_dim = get_required_usize(cfg, "embed_dim")?;
    let _hidden_size = get_required_usize(cfg, "hidden_size")?;
    let intermediate = get_required_usize(cfg, "intermediate_size")?;
    let num_layers = get_required_usize(cfg, "num_hidden_layers")?;
    let num_heads = get_required_usize(cfg, "num_attention_heads")?;
    let _head_dim = embed_dim
        .checked_div(num_heads)
        .context("embed_dim must be divisible by num_attention_heads")?;
    let use_bias = cfg
        .get("use_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut specs = Vec::new();
    for layer_idx in 0..num_layers {
        let layer_prefix = format!("vision_tower.blocks.{layer_idx}");
        let attn_prefix = format!("{layer_prefix}.attn");
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.qkv.weight"),
            out_dim: embed_dim * 3,
            in_dim: embed_dim,
            bias: bias_name(use_bias, &attn_prefix, "qkv.bias"),
        });
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.proj.weight"),
            out_dim: embed_dim,
            in_dim: embed_dim,
            bias: bias_name(use_bias, &attn_prefix, "proj.bias"),
        });

        let mlp_prefix = format!("{layer_prefix}.mlp");
        specs.push(LinearSpec {
            name: format!("{mlp_prefix}.fc1.weight"),
            out_dim: intermediate,
            in_dim: embed_dim,
            bias: bias_name(use_bias, &mlp_prefix, "fc1.bias"),
        });
        specs.push(LinearSpec {
            name: format!("{mlp_prefix}.fc2.weight"),
            out_dim: embed_dim,
            in_dim: intermediate,
            bias: bias_name(use_bias, &mlp_prefix, "fc2.bias"),
        });
        specs.push(LinearSpec {
            name: format!("{mlp_prefix}.fc3.weight"),
            out_dim: intermediate,
            in_dim: embed_dim,
            bias: bias_name(use_bias, &mlp_prefix, "fc3.bias"),
        });
    }

    Ok(specs)
}

fn merger_specs(cfg: &Map<String, Value>) -> Vec<LinearSpec> {
    let embed_dim = get_optional_nonzero(cfg, "embed_dim").unwrap_or(1536);
    let hidden_size = get_optional_nonzero(cfg, "hidden_size").unwrap_or(embed_dim);
    let merge = get_optional_nonzero(cfg, "spatial_merge_size").unwrap_or(2);
    let group_input = embed_dim
        .checked_mul(merge)
        .and_then(|v| v.checked_mul(merge))
        .unwrap_or(embed_dim * merge * merge);
    let prefix = "vision_tower.merger";
    vec![
        LinearSpec {
            name: format!("{prefix}.mlp.0.weight"),
            out_dim: group_input,
            in_dim: group_input,
            bias: Some(format!("{prefix}.mlp.0.bias")),
        },
        LinearSpec {
            name: format!("{prefix}.mlp.2.weight"),
            out_dim: hidden_size,
            in_dim: group_input,
            bias: Some(format!("{prefix}.mlp.2.bias")),
        },
    ]
}

fn bias_name(enabled: bool, prefix: &str, suffix: &str) -> Option<String> {
    if enabled {
        Some(format!("{prefix}.{suffix}"))
    } else {
        None
    }
}
