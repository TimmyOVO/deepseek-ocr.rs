mod common;

use anyhow::Result;
use common::test_utils::{shared_language_config, shared_transformer_weights};
use deepseek_ocr_infer_deepseek::transformer::weights::MlpWeights;

#[test]
fn transformer_weights_load_from_safetensor() -> Result<()> {
    let cfg = shared_language_config()?;
    let weights = shared_transformer_weights()?;
    assert_eq!(
        weights.layers.len(),
        cfg.num_hidden_layers,
        "layer count mismatch",
    );

    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    let first = &weights.layers[0];
    assert_eq!(
        first
            .attention
            .q_proj
            .weight
            .shape()
            .dims2()
            .expect("q_proj dims"),
        (cfg.num_attention_heads * head_dim, cfg.hidden_size,),
    );

    if let Some(routed) = cfg.n_routed_experts {
        match &weights.layers[1].mlp {
            MlpWeights::Dense(_) => panic!("layer 1 should use MoE weights"),
            MlpWeights::Moe(moe) => {
                assert_eq!(moe.experts.len(), routed);
                assert_eq!(
                    moe.gate_weight.shape().dims2().expect("gate dims"),
                    (routed, cfg.hidden_size)
                );
            }
        }
    }
    Ok(())
}
