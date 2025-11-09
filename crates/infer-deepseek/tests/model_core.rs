mod common;

use anyhow::Result;
use candle_core::{DType, Tensor};
use common::test_utils::with_shared_ocr_model;
use deepseek_ocr_infer_deepseek::model::{DeepseekOcrModel, GenerateOptions, VisionInput};

fn with_model<F>(label: &str, f: F) -> Result<()>
where
    F: FnOnce(&DeepseekOcrModel) -> Result<()>,
{
    match with_shared_ocr_model(f) {
        Ok(result) => Ok(result),
        Err(err) => {
            eprintln!("skipping {label}: {err}");
            Ok(())
        }
    }
}

#[test]
fn cache_matches_language_layers() -> Result<()> {
    with_model("DeepseekOcrModel cache test", |model| {
        let cache = model.new_cache();
        assert_eq!(
            cache.num_layers(),
            model.language_model().transformer_weights().layers.len()
        );
        Ok(())
    })
}

#[test]
fn forward_supports_image_mask() -> Result<()> {
    with_model("DeepseekOcrModel image forward test", |model| {
        let device = model.device().clone();
        let input_ids = Tensor::zeros((1, 4), DType::I64, &device)?;
        let mask = Tensor::from_vec(vec![0u8, 1, 1, 0], (1, 4), &device)?;
        let mut cache = model.new_cache();
        let mut guard = model.prompt_guard(&mut cache);
        let output = model.forward(
            Some(&input_ids),
            None,
            None,
            None,
            Some(&mask),
            None,
            None,
            Some(guard.cache()),
            true,
        )?;
        assert_eq!(
            output.logits.shape().dims3()?,
            (1, 4, model.language_model().config().vocab_size)
        );
        Ok(())
    })
}

#[test]
fn inject_image_tokens_uses_supplied_embeddings() -> Result<()> {
    with_model("DeepseekOcrModel injection test", |model| {
        let device = model.device().clone();
        let hidden = model.language_model().config().hidden_size;
        let embeddings = Tensor::zeros((1, 3, hidden), model.dtype(), &device)?.contiguous()?;
        let mask = Tensor::from_vec(vec![0u8, 1, 0], (1, 3), &device)?;
        let expected: Vec<f32> = (0..hidden).map(|i| i as f32).collect();
        let replacement = Tensor::from_vec(expected.clone(), (1, hidden), &device)?.contiguous()?;
        let tokens = vec![replacement];
        let updated =
            model.inject_image_tokens_for_tests(embeddings, &mask, Some(tokens.as_slice()))?;
        let inserted = updated
            .get(0)?
            .get(1)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        assert_eq!(inserted.len(), hidden);
        for (actual, expected) in inserted.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        Ok(())
    })
}

#[test]
fn generate_respects_max_tokens() -> Result<()> {
    with_model("DeepseekOcrModel generate test", |model| {
        let device = model.device().clone();
        let input_ids = Tensor::zeros((1, 4), DType::I64, &device)?;
        let mask = Tensor::from_vec(vec![0u8, 1, 0, 0], (1, 4), &device)?;
        let mut opts = GenerateOptions::new(3);
        opts.images_seq_mask = Some(&mask);
        opts.eos_token_id = model.language_model().config().eos_token_id;
        let generated = model.generate(&input_ids, opts)?;
        let (_batch, new_tokens) = generated.shape().dims2()?;
        assert!(new_tokens <= 3);
        Ok(())
    })
}

#[test]
fn compute_image_embeddings_produces_tokens() -> Result<()> {
    with_model("vision embedding test", |model| {
        let device = model.device().clone();
        let dtype = model.dtype();

        let global = Tensor::zeros((1, 3, 1024, 1024), dtype, &device)?;
        let vision_input = VisionInput {
            global: &global,
            patches: None,
            crop_shape: None,
        };
        let embeddings = model.compute_image_embeddings(&[Some(vision_input)])?;
        assert_eq!(embeddings.len(), 1);
        let (tokens, hidden) = embeddings[0].shape().dims2()?;
        assert!(tokens > 0, "vision projector should yield tokens");
        assert_eq!(
            hidden,
            model.language_model().config().hidden_size,
            "projector output hidden size mismatch"
        );

        let mask = Tensor::from_vec(vec![1u8; tokens], (1, tokens), &device)?;
        let base = Tensor::zeros((1, tokens, hidden), dtype, &device)?.contiguous()?;
        let updated =
            model.inject_image_tokens_for_tests(base, &mask, Some(embeddings.as_slice()))?;
        let updated_rows = updated.get(0)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let expected_rows = embeddings[0].to_dtype(DType::F32)?.to_vec2::<f32>()?;
        assert_eq!(updated_rows.len(), expected_rows.len());
        for (row_a, row_b) in updated_rows.iter().zip(expected_rows.iter()) {
            assert_eq!(row_a.len(), row_b.len());
            for (a, b) in row_a.iter().zip(row_b.iter()) {
                assert!((a - b).abs() < 1e-4);
            }
        }
        Ok(())
    })
}

#[test]
fn vision_inputs_and_precomputed_embeddings_align() -> Result<()> {
    with_model("vision alignment test", |model| {
        let device = model.device().clone();
        let dtype = model.dtype();

        let global = Tensor::zeros((1, 3, 1024, 1024), dtype, &device)?;
        let vision_spec = VisionInput {
            global: &global,
            patches: None,
            crop_shape: None,
        };
        let vision_inputs = vec![Some(vision_spec)];
        let cached_embeddings = model.compute_image_embeddings(&vision_inputs)?;
        assert_eq!(cached_embeddings.len(), 1);
        let (tokens, hidden) = cached_embeddings[0].shape().dims2()?;
        let input_ids = Tensor::zeros((1, tokens), DType::I64, &device)?;
        let mask = Tensor::from_vec(vec![1u8; tokens], (1, tokens), &device)?;

        let mut cache_inputs = model.new_cache();
        let mut guard_inputs = model.prompt_guard(&mut cache_inputs);
        let via_inputs = model.forward(
            Some(&input_ids),
            None,
            None,
            None,
            Some(&mask),
            Some(vision_inputs.as_slice()),
            None,
            Some(guard_inputs.cache()),
            true,
        )?;

        let mut cache_embeds = model.new_cache();
        let mut guard_embeds = model.prompt_guard(&mut cache_embeds);
        let via_embeddings = model.forward(
            Some(&input_ids),
            None,
            None,
            None,
            Some(&mask),
            None,
            Some(cached_embeddings.as_slice()),
            Some(guard_embeds.cache()),
            true,
        )?;

        let lhs = via_inputs.logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let rhs = via_embeddings
            .logits
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?;
        assert_eq!(lhs.len(), rhs.len());
        for (lhs_batch, rhs_batch) in lhs.iter().zip(rhs.iter()) {
            assert_eq!(lhs_batch.len(), rhs_batch.len());
            for (lhs_step, rhs_step) in lhs_batch.iter().zip(rhs_batch.iter()) {
                assert_eq!(lhs_step.len(), rhs_step.len());
                for (l, r) in lhs_step.iter().zip(rhs_step.iter()) {
                    assert!((l - r).abs() < 1e-4);
                }
            }
        }
        assert_eq!(hidden, model.language_model().config().hidden_size);
        Ok(())
    })
}
