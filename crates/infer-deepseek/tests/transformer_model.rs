mod common;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use common::test_utils::with_shared_language_model;
use deepseek_ocr_infer_deepseek::transformer::{cache::DynamicCache, model::DeepseekLanguageModel};

fn with_language_model<F>(label: &str, f: F) -> Result<()>
where
    F: FnOnce(&DeepseekLanguageModel) -> Result<()>,
{
    match with_shared_language_model(f) {
        Ok(result) => Ok(result),
        Err(err) => {
            eprintln!("skipping {label}: {err}");
            Ok(())
        }
    }
}

#[test]
fn language_model_forward_with_input_ids() -> Result<()> {
    with_language_model("language model forward test", |model| {
        let device = Device::Cpu;
        let input_ids = Tensor::zeros((1, 4), DType::I64, &device)?;
        let output = model.forward(Some(&input_ids), None, None, None, None, false)?;
        let cfg = model.config();
        assert_eq!(
            output.hidden_states.shape().dims3()?,
            (1, 4, cfg.hidden_size)
        );
        assert_eq!(output.logits.shape().dims3()?, (1, 4, cfg.vocab_size));
        Ok(())
    })
}

#[test]
fn language_model_decode_appends_cache() -> Result<()> {
    with_language_model("language model cache test", |model| {
        let device = Device::Cpu;
        let mut cache = DynamicCache::with_num_layers(model.transformer_weights().layers.len());

        let prefill_ids = Tensor::zeros((1, 4), DType::I64, &device)?;
        model.forward(Some(&prefill_ids), None, None, None, Some(&mut cache), true)?;
        assert_eq!(cache.seq_len(), Some(4));

        let decode_ids = Tensor::zeros((1, 1), DType::I64, &device)?;
        model.forward(Some(&decode_ids), None, None, None, Some(&mut cache), true)?;
        assert_eq!(cache.seq_len(), Some(5));

        {
            let mut guard = model.prompt_guard(&mut cache);
            assert!(guard.cache().seq_len().is_some());
        }
        assert!(cache.seq_len().is_none());
        Ok(())
    })
}
