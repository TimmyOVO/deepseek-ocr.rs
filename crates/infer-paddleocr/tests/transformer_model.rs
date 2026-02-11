use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use deepseek_ocr_infer_paddleocr::{
    config::PaddleOcrVlConfig,
    transformer::{
        ErnieAttentionWeights, ErnieDecoder, ErnieDecoderLayerWeights, ErnieMlpWeights,
        ErnieModelWeights, LinearWeights,
    },
};
use serde_json::json;

fn test_config() -> PaddleOcrVlConfig {
    let cfg_json = json!({
        "head_dim": 6,
        "hidden_size": 18,
        "intermediate_size": 24,
        "vocab_size": 32,
        "num_attention_heads": 3,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "rope_scaling": { "mrope_section": [1,1,1], "type": "default" },
        "hidden_act": "silu",
        "use_bias": false,
        "use_cache": true,
        "use_flash_attention": false,
        "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_channels": 3,
            "image_size": 32,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "tokens_per_second": 1
        }
    });
    serde_json::from_value(cfg_json).expect("valid config json")
}

fn linear(out_dim: usize, in_dim: usize, device: &Device) -> anyhow::Result<LinearWeights> {
    Ok(LinearWeights {
        weight: Some(Tensor::zeros((out_dim, in_dim), DType::F32, device)?),
        bias: None,
        qmatmul: None,
        out_dim,
        in_dim,
        label: format!("test.linear.{out_dim}x{in_dim}"),
    })
}

fn layer_weights(cfg: &PaddleOcrVlConfig, device: &Device) -> anyhow::Result<ErnieDecoderLayerWeights> {
    let attn = ErnieAttentionWeights {
        q_proj: linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, device)?,
        k_proj: linear(
            cfg.resolved_num_key_value_heads() * cfg.head_dim,
            cfg.hidden_size,
            device,
        )?,
        v_proj: linear(
            cfg.resolved_num_key_value_heads() * cfg.head_dim,
            cfg.hidden_size,
            device,
        )?,
        o_proj: linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, device)?,
    };
    let mlp = ErnieMlpWeights {
        gate_proj: linear(cfg.intermediate_size, cfg.hidden_size, device)?,
        up_proj: linear(cfg.intermediate_size, cfg.hidden_size, device)?,
        down_proj: linear(cfg.hidden_size, cfg.intermediate_size, device)?,
    };
    let norm = Tensor::ones(cfg.hidden_size, DType::F32, device)?;
    Ok(ErnieDecoderLayerWeights {
        attention: attn,
        mlp,
        input_layernorm: norm.clone(),
        post_attention_layernorm: norm,
    })
}

#[test]
fn decoder_forward_shapes() -> anyhow::Result<()> {
    let cfg = Arc::new(test_config());
    let device = Device::Cpu;
    let embed = Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F32, &device)?;
    let layers = (0..cfg.num_hidden_layers)
        .map(|_| layer_weights(cfg.as_ref(), &device))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let final_norm = Tensor::ones(cfg.hidden_size, DType::F32, &device)?;
    let weights = ErnieModelWeights {
        embed_tokens: embed,
        layers,
        final_norm,
    };
    let lm_head = LinearWeights {
        weight: Some(Tensor::zeros(
            (cfg.vocab_size, cfg.hidden_size),
            DType::F32,
            &device,
        )?),
        bias: None,
        qmatmul: None,
        out_dim: cfg.vocab_size,
        in_dim: cfg.hidden_size,
        label: "tests.lm_head.weight".to_string(),
    };
    let decoder = ErnieDecoder::from_parts(Arc::clone(&cfg), weights, lm_head)?;

    let input_ids = Tensor::zeros((2, 4), DType::I64, &device)?;
    let output = decoder.forward(Some(&input_ids), None, None, None, None, false)?;
    assert_eq!(output.hidden_states.shape().dims(), [2, 4, cfg.hidden_size]);
    assert_eq!(output.logits.shape().dims(), [2, 4, cfg.vocab_size]);

    let mut cache = decoder.new_cache();
    decoder.forward(Some(&input_ids), None, None, None, Some(&mut cache), true)?;
    let next = Tensor::zeros((2, 1), DType::I64, &device)?;
    decoder.forward(Some(&next), None, None, None, Some(&mut cache), true)?;
    Ok(())
}
