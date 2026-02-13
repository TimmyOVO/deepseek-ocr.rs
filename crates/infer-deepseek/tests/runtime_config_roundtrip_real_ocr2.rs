use anyhow::Result;
use deepseek_ocr_core::config::DeepseekRuntimeConfig;
use deepseek_ocr_infer_deepseek::config::DeepseekV2Config;

mod common;

#[test]
fn runtime_config_roundtrip_real_ocr2_config() -> Result<()> {
    let cfg = common::load_fixture("deepseek_ocr2_config.json")?;
    let runtime1 = cfg.resolved_runtime_config()?;

    let legacy: DeepseekV2Config = runtime1.clone().into();
    let runtime2 = DeepseekRuntimeConfig::try_from(legacy)?;

    assert_eq!(runtime1.base.hidden_size, runtime2.base.hidden_size);
    assert_eq!(
        runtime1.base.num_attention_heads,
        runtime2.base.num_attention_heads
    );
    assert_eq!(runtime1.base.num_kv_heads, runtime2.base.num_kv_heads);
    assert_eq!(runtime1.base.kv_head_dim, runtime2.base.kv_head_dim);
    assert_eq!(runtime1.base.v_head_dim, runtime2.base.v_head_dim);
    assert_eq!(
        runtime1.base.max_position_embeddings,
        runtime2.base.max_position_embeddings
    );
    assert_eq!(runtime1.topk_method, runtime2.topk_method);
    assert_eq!(runtime1.scoring_func, runtime2.scoring_func);
    assert_eq!(runtime1.n_routed_experts, runtime2.n_routed_experts);
    assert_eq!(runtime1.num_experts_per_tok, runtime2.num_experts_per_tok);
    Ok(())
}
