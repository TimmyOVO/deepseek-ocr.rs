use deepseek_ocr_dsq_models::{AdapterRegistry, AdapterScope};
use serde_json::json;

#[test]
fn deepseek_adapter_discovers_projector_and_lm_head() {
    let cfg = json!({
        "model_type": "deepseek_vl_v2",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "n_routed_experts": 0,
        "n_shared_experts": 0,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 0,
        "lm_head": true,
        "vocab_size": 32,
        "projector_config": {
            "n_embed": 8,
            "input_dim": 4
        }
    });
    let adapter = AdapterRegistry::global()
        .infer_adapter(&cfg)
        .expect("infer adapter");
    assert_eq!(adapter.id(), "deepseek-ocr");
    let specs = adapter
        .discover(&cfg, AdapterScope::TextAndProjector)
        .expect("discover specs");
    assert!(specs.iter().any(|spec| spec.name == "lm_head.weight"));
    assert!(specs
        .iter()
        .any(|spec| spec.name == "model.projector.layers.weight"));
}

#[test]
fn paddle_adapter_detects_from_model_type() {
    let cfg = json!({
        "model_type": "paddleocr_vl",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "vocab_size": 32000,
        "vision_config": {
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 8
        }
    });
    let adapter = AdapterRegistry::global()
        .infer_adapter(&cfg)
        .expect("infer adapter");
    assert_eq!(adapter.id(), "paddleocr-vl");
    let specs = adapter
        .discover(&cfg, AdapterScope::TextAndProjector)
        .expect("discover specs");
    assert!(specs.iter().any(|spec| {
        spec.name
            .contains("visual.vision_model.encoder.layers.0.self_attn.q_proj")
    }));
}

#[test]
fn dots_adapter_detects_from_model_type() {
    let cfg = json!({
        "model_type": "dots_ocr",
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_hidden_layers": 2,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "vocab_size": 32000,
        "attention_bias": true,
        "vision_config": {
            "embed_dim": 1536,
            "hidden_size": 1536,
            "intermediate_size": 4224,
            "num_hidden_layers": 2,
            "num_attention_heads": 12,
            "num_channels": 3,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "rms_norm_eps": 1e-5,
            "use_bias": false,
            "attn_implementation": "flash_attention_2",
            "init_merger_std": 0.02,
            "initializer_range": 0.02,
            "is_causal": false,
            "post_norm": true
        }
    });
    let adapter = AdapterRegistry::global()
        .infer_adapter(&cfg)
        .expect("infer adapter");
    assert_eq!(adapter.id(), "dots-ocr");
    let specs = adapter
        .discover(&cfg, AdapterScope::TextAndProjector)
        .expect("discover specs");
    assert!(specs
        .iter()
        .any(|spec| spec.name == "model.layers.0.self_attn.q_proj.weight"));
    assert!(specs
        .iter()
        .any(|spec| spec.name == "vision_tower.blocks.0.attn.qkv.weight"));
    assert!(specs.iter().any(|spec| spec.name == "lm_head.weight"));
}
