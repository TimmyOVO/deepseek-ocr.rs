use anyhow::{Context, Result};
use deepseek_ocr_dsq::DsqTensorDType;
use serde_json::{Map, Value};

use crate::{
    adapters::helpers::{get_optional_nonzero, get_optional_usize, get_required_usize},
    AdapterScope, LinearSpec, ModelAdapter, QuantContext,
};

pub struct DeepSeekOcrAdapter;

impl ModelAdapter for DeepSeekOcrAdapter {
    fn id(&self) -> &'static str {
        "deepseek-ocr"
    }

    fn supports(&self, cfg: &Value) -> bool {
        let root = match cfg.as_object() {
            Some(obj) => obj,
            None => return false,
        };
        let lang_cfg = root
            .get("language_config")
            .and_then(Value::as_object)
            .or_else(|| {
                root.get("model_type")
                    .and_then(Value::as_str)
                    .filter(|ty| ty.to_ascii_lowercase().contains("deepseek"))
                    .map(|_| root)
            });
        match lang_cfg {
            Some(lang) => {
                lang.contains_key("hidden_size")
                    && lang.contains_key("num_hidden_layers")
                    && lang.contains_key("num_attention_heads")
            }
            None => false,
        }
    }

    fn discover(&self, cfg: &Value, scope: AdapterScope) -> Result<Vec<LinearSpec>> {
        let root = cfg
            .as_object()
            .context("config JSON must contain a top-level object")?;
        let lang_cfg = language_config(cfg).context("missing language configuration")?;
        let hidden_size = get_required_usize(lang_cfg, "hidden_size")?;
        let num_layers = get_required_usize(lang_cfg, "num_hidden_layers")?;
        let num_heads = get_required_usize(lang_cfg, "num_attention_heads")?;
        let num_kv_heads = get_optional_usize(lang_cfg, "num_key_value_heads").unwrap_or(num_heads);
        let head_dim = hidden_size
            .checked_div(num_heads)
            .context("hidden_size must be divisible by num_attention_heads")?;
        let v_head_dim = get_optional_nonzero(lang_cfg, "v_head_dim").unwrap_or(head_dim);
        let intermediate_size = get_required_usize(lang_cfg, "intermediate_size")?;
        let moe_intermediate =
            get_optional_nonzero(lang_cfg, "moe_intermediate_size").unwrap_or(intermediate_size);
        let n_routed = get_optional_usize(lang_cfg, "n_routed_experts").unwrap_or(0);
        let n_shared = get_optional_usize(lang_cfg, "n_shared_experts").unwrap_or(0);
        let moe_layer_freq = get_optional_nonzero(lang_cfg, "moe_layer_freq").unwrap_or(1);
        let first_dense = get_optional_usize(lang_cfg, "first_k_dense_replace").unwrap_or(0);
        let vocab_size = get_required_usize(lang_cfg, "vocab_size")?;
        let lm_head_enabled = lang_cfg
            .get("lm_head")
            .and_then(Value::as_bool)
            .unwrap_or(true);

        let mut specs = Vec::new();
        for layer_idx in 0..num_layers {
            let layer_prefix = format!("model.layers.{layer_idx}");
            let attn_prefix = format!("{layer_prefix}.self_attn");
            specs.push(LinearSpec {
                name: format!("{attn_prefix}.q_proj.weight"),
                out_dim: num_heads * head_dim,
                in_dim: hidden_size,
                bias: Some(format!("{attn_prefix}.q_proj.bias")),
            });
            specs.push(LinearSpec {
                name: format!("{attn_prefix}.k_proj.weight"),
                out_dim: num_kv_heads * head_dim,
                in_dim: hidden_size,
                bias: Some(format!("{attn_prefix}.k_proj.bias")),
            });
            specs.push(LinearSpec {
                name: format!("{attn_prefix}.v_proj.weight"),
                out_dim: num_kv_heads * v_head_dim,
                in_dim: hidden_size,
                bias: Some(format!("{attn_prefix}.v_proj.bias")),
            });
            specs.push(LinearSpec {
                name: format!("{attn_prefix}.o_proj.weight"),
                out_dim: hidden_size,
                in_dim: num_heads * v_head_dim,
                bias: Some(format!("{attn_prefix}.o_proj.bias")),
            });
            let mlp_prefix = format!("{layer_prefix}.mlp");
            if should_use_moe(layer_idx, n_routed, moe_layer_freq, first_dense) {
                for expert_idx in 0..n_routed {
                    let expert_prefix = format!("{mlp_prefix}.experts.{expert_idx}");
                    specs.extend(mlp_specs(&expert_prefix, hidden_size, moe_intermediate));
                }
                if n_shared > 0 {
                    let shared_prefix = format!("{mlp_prefix}.shared_experts");
                    specs.extend(mlp_specs(
                        &shared_prefix,
                        hidden_size,
                        moe_intermediate * n_shared,
                    ));
                }
            } else {
                specs.extend(mlp_specs(&mlp_prefix, hidden_size, intermediate_size));
            }
        }

        if lm_head_enabled {
            specs.push(LinearSpec {
                name: "lm_head.weight".to_string(),
                out_dim: vocab_size,
                in_dim: hidden_size,
                bias: None,
            });
        }

        if scope.includes_projector() {
            let projector = root
                .get("projector_config")
                .and_then(Value::as_object)
                .context("projector_config missing from config")?;
            let out_dim = get_required_usize(projector, "n_embed")?;
            let in_dim = get_required_usize(projector, "input_dim")?;
            specs.push(LinearSpec {
                name: "model.projector.layers.weight".to_string(),
                out_dim,
                in_dim,
                bias: Some("model.projector.layers.bias".to_string()),
            });
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
        match tensor {
            "lm_head.weight" | "model.projector.layers.weight" => Some(DsqTensorDType::Q8_0),
            _ => None,
        }
    }
}

fn language_config(cfg: &Value) -> Option<&Map<String, Value>> {
    let root = cfg.as_object()?;
    root.get("language_config")
        .and_then(Value::as_object)
        .or(Some(root))
}

fn should_use_moe(layer_idx: usize, n_routed: usize, freq: usize, first_dense: usize) -> bool {
    if n_routed == 0 {
        return false;
    }
    if layer_idx < first_dense {
        return false;
    }
    freq != 0 && (layer_idx % freq == 0)
}

fn mlp_specs(prefix: &str, hidden_size: usize, intermediate_size: usize) -> Vec<LinearSpec> {
    vec![
        LinearSpec {
            name: format!("{prefix}.gate_proj.weight"),
            out_dim: intermediate_size,
            in_dim: hidden_size,
            bias: Some(format!("{prefix}.gate_proj.bias")),
        },
        LinearSpec {
            name: format!("{prefix}.up_proj.weight"),
            out_dim: intermediate_size,
            in_dim: hidden_size,
            bias: Some(format!("{prefix}.up_proj.bias")),
        },
        LinearSpec {
            name: format!("{prefix}.down_proj.weight"),
            out_dim: hidden_size,
            in_dim: intermediate_size,
            bias: Some(format!("{prefix}.down_proj.bias")),
        },
    ]
}
