use anyhow::{bail, Context, Result};
use deepseek_ocr_dsq::DsqTensorDType;
use serde_json::{Map, Value};

use crate::{
    adapters::helpers::{
        get_optional_nonzero, get_optional_usize, get_required_usize, root_object,
    },
    AdapterScope, LinearSpec, ModelAdapter, QuantContext,
};

pub struct PaddleOcrVlAdapter;

impl ModelAdapter for PaddleOcrVlAdapter {
    fn id(&self) -> &'static str {
        "paddleocr-vl"
    }

    fn supports(&self, cfg: &Value) -> bool {
        cfg.as_object()
            .and_then(|root| root.get("model_type"))
            .and_then(Value::as_str)
            .map(|ty| ty.eq_ignore_ascii_case("paddleocr_vl"))
            .unwrap_or(false)
    }

    fn discover(&self, cfg: &Value, scope: AdapterScope) -> Result<Vec<LinearSpec>> {
        let root = root_object(cfg)?;
        let mut specs = text_decoder_specs(root)?;
        if scope.includes_projector() {
            if let Some(vision) = root.get("vision_config").and_then(Value::as_object) {
                specs.extend(vision_encoder_specs(vision)?);
                specs.extend(bridge_specs(root, vision));
            } else {
                bail!("vision_config missing from PaddleOCR-VL configuration");
            }
        }
        Ok(specs)
    }

    fn recommend_dtype(
        &self,
        tensor: &str,
        _in_dim: usize,
        ctx: &QuantContext,
    ) -> Option<DsqTensorDType> {
        if ctx.primary == DsqTensorDType::Q8_0 {
            return None;
        }
        if tensor == "lm_head.weight" {
            return Some(DsqTensorDType::Q8_0);
        }
        None
    }
}

fn text_decoder_specs(root: &Map<String, Value>) -> Result<Vec<LinearSpec>> {
    let hidden_size = get_required_usize(root, "hidden_size")?;
    let num_layers = get_required_usize(root, "num_hidden_layers")?;
    let num_heads = get_required_usize(root, "num_attention_heads")?;
    let num_kv_heads = get_optional_usize(root, "num_key_value_heads").unwrap_or(num_heads);
    let head_dim = get_optional_usize(root, "head_dim").unwrap_or_else(|| hidden_size / num_heads);
    let k_head_dim = head_dim;
    let v_head_dim = get_optional_usize(root, "v_head_dim").unwrap_or(k_head_dim);
    let intermediate = get_required_usize(root, "intermediate_size")?;
    let vocab_size = get_required_usize(root, "vocab_size")?;
    let use_bias = root
        .get("use_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut specs = Vec::new();
    for layer_idx in 0..num_layers {
        let layer_prefix = format!("model.layers.{layer_idx}");
        let attn_prefix = format!("{layer_prefix}.self_attn");
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.q_proj.weight"),
            out_dim: num_heads * head_dim,
            in_dim: hidden_size,
            bias: bias_name(use_bias, &attn_prefix, "q_proj.bias"),
        });
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.k_proj.weight"),
            out_dim: num_kv_heads * k_head_dim,
            in_dim: hidden_size,
            bias: bias_name(use_bias, &attn_prefix, "k_proj.bias"),
        });
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.v_proj.weight"),
            out_dim: num_kv_heads * v_head_dim,
            in_dim: hidden_size,
            bias: bias_name(use_bias, &attn_prefix, "v_proj.bias"),
        });
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.o_proj.weight"),
            out_dim: hidden_size,
            in_dim: num_heads * head_dim,
            bias: bias_name(use_bias, &attn_prefix, "o_proj.bias"),
        });

        let mlp_prefix = format!("{layer_prefix}.mlp");
        specs.push(LinearSpec {
            name: format!("{mlp_prefix}.gate_proj.weight"),
            out_dim: intermediate,
            in_dim: hidden_size,
            bias: bias_name(use_bias, &mlp_prefix, "gate_proj.bias"),
        });
        specs.push(LinearSpec {
            name: format!("{mlp_prefix}.up_proj.weight"),
            out_dim: intermediate,
            in_dim: hidden_size,
            bias: bias_name(use_bias, &mlp_prefix, "up_proj.bias"),
        });
        specs.push(LinearSpec {
            name: format!("{mlp_prefix}.down_proj.weight"),
            out_dim: hidden_size,
            in_dim: intermediate,
            bias: bias_name(use_bias, &mlp_prefix, "down_proj.bias"),
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
    let hidden_size = get_required_usize(cfg, "hidden_size")?;
    let intermediate = get_required_usize(cfg, "intermediate_size")?;
    let num_layers = get_required_usize(cfg, "num_hidden_layers")?;
    let num_heads = get_required_usize(cfg, "num_attention_heads")?;
    let _head_dim = hidden_size
        .checked_div(num_heads)
        .context("vision hidden_size must be divisible by num_attention_heads")?;

    let mut specs = Vec::new();
    for layer_idx in 0..num_layers {
        let layer_prefix = format!("visual.vision_model.encoder.layers.{layer_idx}");
        let attn_prefix = format!("{layer_prefix}.self_attn");
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.q_proj.weight"),
            out_dim: hidden_size,
            in_dim: hidden_size,
            bias: Some(format!("{attn_prefix}.q_proj.bias")),
        });
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.k_proj.weight"),
            out_dim: hidden_size,
            in_dim: hidden_size,
            bias: Some(format!("{attn_prefix}.k_proj.bias")),
        });
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.v_proj.weight"),
            out_dim: hidden_size,
            in_dim: hidden_size,
            bias: Some(format!("{attn_prefix}.v_proj.bias")),
        });
        specs.push(LinearSpec {
            name: format!("{attn_prefix}.out_proj.weight"),
            out_dim: hidden_size,
            in_dim: hidden_size,
            bias: Some(format!("{attn_prefix}.out_proj.bias")),
        });

        let mlp_prefix = format!("{layer_prefix}.mlp");
        specs.push(LinearSpec {
            name: format!("{mlp_prefix}.fc1.weight"),
            out_dim: intermediate,
            in_dim: hidden_size,
            bias: Some(format!("{mlp_prefix}.fc1.bias")),
        });
        specs.push(LinearSpec {
            name: format!("{mlp_prefix}.fc2.weight"),
            out_dim: hidden_size,
            in_dim: intermediate,
            bias: Some(format!("{mlp_prefix}.fc2.bias")),
        });
    }

    specs.push(LinearSpec {
        name: "visual.vision_model.head.attention.in_proj_weight".to_string(),
        out_dim: hidden_size * 3,
        in_dim: hidden_size,
        bias: Some("visual.vision_model.head.attention.in_proj_bias".to_string()),
    });
    specs.push(LinearSpec {
        name: "visual.vision_model.head.attention.out_proj.weight".to_string(),
        out_dim: hidden_size,
        in_dim: hidden_size,
        bias: Some("visual.vision_model.head.attention.out_proj.bias".to_string()),
    });
    specs.push(LinearSpec {
        name: "visual.vision_model.head.mlp.fc1.weight".to_string(),
        out_dim: intermediate,
        in_dim: hidden_size,
        bias: Some("visual.vision_model.head.mlp.fc1.bias".to_string()),
    });
    specs.push(LinearSpec {
        name: "visual.vision_model.head.mlp.fc2.weight".to_string(),
        out_dim: hidden_size,
        in_dim: intermediate,
        bias: Some("visual.vision_model.head.mlp.fc2.bias".to_string()),
    });

    Ok(specs)
}

fn bridge_specs(root: &Map<String, Value>, vision: &Map<String, Value>) -> Vec<LinearSpec> {
    let text_dim = get_optional_usize(root, "hidden_size").unwrap_or(1024);
    let vision_dim = get_optional_usize(vision, "hidden_size").unwrap_or(1152);
    let expanded = get_optional_nonzero(root, "bridge_intermediate").unwrap_or(vision_dim * 4);
    vec![
        LinearSpec {
            name: "mlp_AR.linear_1.weight".to_string(),
            out_dim: expanded,
            in_dim: expanded,
            bias: Some("mlp_AR.linear_1.bias".to_string()),
        },
        LinearSpec {
            name: "mlp_AR.linear_2.weight".to_string(),
            out_dim: text_dim,
            in_dim: expanded,
            bias: Some("mlp_AR.linear_2.bias".to_string()),
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
