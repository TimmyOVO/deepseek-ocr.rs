use anyhow::Result;
use deepseek_ocr_core::config::RopeConfig;

mod common;

#[test]
fn runtime_config_from_real_ocr1_config() -> Result<()> {
    let cfg = common::load_fixture("deepseek_ocr1_config.json")?;
    let runtime = cfg.resolved_runtime_config()?;

    assert_eq!(runtime.base.hidden_size, 1280);
    assert_eq!(runtime.base.num_attention_heads, 10);
    assert_eq!(runtime.base.num_kv_heads, 10);
    assert_eq!(runtime.base.head_dim, 128);
    assert_eq!(runtime.base.kv_head_dim, 128);
    assert_eq!(runtime.base.v_head_dim, 128);
    assert_eq!(runtime.base.max_position_embeddings, 8192);
    assert_eq!(runtime.topk_method, "greedy");
    assert_eq!(runtime.scoring_func, "softmax");
    assert_eq!(runtime.first_k_dense_replace, 1);
    assert_eq!(runtime.n_routed_experts, 64);
    assert_eq!(runtime.num_experts_per_tok, Some(6));
    match runtime.base.rope {
        RopeConfig::Standard { rotary_dim, .. } => assert_eq!(rotary_dim, 128),
        RopeConfig::MultiModal { .. } => unreachable!("deepseek should use standard rope"),
    }
    Ok(())
}
