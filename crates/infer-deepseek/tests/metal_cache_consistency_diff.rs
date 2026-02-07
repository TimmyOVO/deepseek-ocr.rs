mod common;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor, shape::D};
use deepseek_ocr_infer_deepseek::model::DeepseekOcrModel;
use serde::Deserialize;
use std::fs;

use common::test_utils::workspace_path;

#[derive(Debug, Deserialize)]
struct BaselineMetadata {
    #[serde(default)]
    model: Option<String>,
    image: String,
    #[serde(default)]
    base_size: Option<u32>,
    #[serde(default)]
    image_size: Option<u32>,
    #[serde(default)]
    crop_mode: Option<bool>,
    #[serde(default)]
    prompt_assets_path: Option<String>,
    #[serde(default)]
    output_tokens_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PromptAssets {
    input_ids: Vec<i64>,
    images_seq_mask: Vec<u8>,
}

#[derive(Debug, Deserialize)]
struct OutputTokens {
    tokens: Vec<i64>,
    prefill_len: usize,
    #[serde(default)]
    eos_token_id: Option<i64>,
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let b = b.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let a_vec = a.flatten_all()?.to_vec1::<f32>()?;
    let b_vec = b.flatten_all()?.to_vec1::<f32>()?;
    ensure!(
        a_vec.len() == b_vec.len(),
        "diff shape mismatch: {} vs {}",
        a_vec.len(),
        b_vec.len()
    );
    let mut max_abs = 0f32;
    for (av, bv) in a_vec.iter().zip(b_vec.iter()) {
        let diff = (av - bv).abs();
        if diff > max_abs {
            max_abs = diff;
        }
    }
    Ok(max_abs)
}

#[test]
#[ignore]
fn metal_cache_consistency_diff() -> Result<()> {
    let case_name = std::env::var("DEEPSEEK_OCR_METAL_DIFF_CASE")
        .unwrap_or_else(|_| "ocr2__test2_png__grounding_md__8192_ng20".to_string());
    let case_dir = workspace_path(format!("baselines/long/{case_name}"));
    let baseline_path = case_dir.join("baseline.json");
    let baseline: BaselineMetadata = serde_json::from_str(&fs::read_to_string(&baseline_path)?)?;

    let prompt_path = baseline
        .prompt_assets_path
        .as_deref()
        .map(workspace_path)
        .unwrap_or_else(|| case_dir.join("prompt.json"));
    let prompt: PromptAssets = serde_json::from_str(&fs::read_to_string(&prompt_path)?)?;

    let output_path = baseline
        .output_tokens_path
        .as_deref()
        .map(workspace_path)
        .unwrap_or_else(|| case_dir.join("output_tokens.json"));
    let output: OutputTokens = serde_json::from_str(&fs::read_to_string(&output_path)?)?;

    let mismatch_idx: usize = std::env::var("DEEPSEEK_OCR_METAL_DIFF_MISMATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(11);

    let use_cache = true;
    let prefill_len = output.prefill_len;
    let mut expected = output.tokens[prefill_len..].to_vec();
    if let Some(eos) = output.eos_token_id
        && expected.last().copied() == Some(eos) {
        expected.pop();
    }
    ensure!(
        mismatch_idx < expected.len(),
        "mismatch index {} exceeds expected length {}",
        mismatch_idx,
        expected.len()
    );

    let prefix_len = prefill_len + mismatch_idx;
    let prefix_tokens = output.tokens[..prefix_len].to_vec();

    let mut mask_full = prompt.images_seq_mask.clone();
    if mask_full.len() < prefix_len {
        mask_full.extend(std::iter::repeat_n(0u8, prefix_len - mask_full.len()));
    } else {
        mask_full.truncate(prefix_len);
    }

    let image_path = workspace_path(&baseline.image);
    let image = image::open(&image_path)
        .with_context(|| format!("failed to open image at {}", image_path.display()))?;

    let base_size = baseline.base_size.unwrap_or(1024);
    let image_size = baseline.image_size.unwrap_or(640);
    let crop_mode = baseline.crop_mode.unwrap_or(true);

    let model_dir = baseline.model.as_deref().unwrap_or("DeepSeek-OCR-2");
    let config_path = workspace_path(format!("{model_dir}/config.json"));
    let weights_path = workspace_path(format!("{model_dir}/model-00001-of-000001.safetensors"));

    let device = Device::new_metal(0).context("failed to init metal device")?;
    let model = DeepseekOcrModel::load(
        Some(&config_path),
        Some(&weights_path),
        None,
        device.clone(),
        DType::F16,
    )?;

    let vision = model.prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)?;
    let image_embeddings = model.compute_image_embeddings(&[Some(vision.as_ref())])?;
    let image_embeddings_slice = image_embeddings.as_slice();

    // Full-prefix cache (single forward).
    let mut cache_full = model.new_cache_for_dtype(DType::F16)?;
    cache_full.clear();
    let input_ids_full =
        Tensor::from_vec(prefix_tokens.clone(), (1, prefix_len), &device)?.to_dtype(DType::I64)?;
    let mask_tensor_full =
        Tensor::from_vec(mask_full.clone(), (1, prefix_len), &device)?.to_dtype(DType::U8)?;
    let out_full_cache = model.forward(
        Some(&input_ids_full),
        None,
        None,
        None,
        Some(&mask_tensor_full),
        None,
        Some(image_embeddings_slice),
        Some(&mut cache_full),
        use_cache,
    )?;

    // Incremental cache (prefill + per-token decode).
    let mut cache_inc = model.new_cache_for_dtype(DType::F16)?;
    cache_inc.clear();
    let prefill_ids = Tensor::from_vec(prompt.input_ids.clone(), (1, prefill_len), &device)?
        .to_dtype(DType::I64)?;
    let prefill_mask = Tensor::from_vec(prompt.images_seq_mask.clone(), (1, prefill_len), &device)?
        .to_dtype(DType::U8)?;
    model.forward(
        Some(&prefill_ids),
        None,
        None,
        None,
        Some(&prefill_mask),
        None,
        Some(image_embeddings_slice),
        Some(&mut cache_inc),
        use_cache,
    )?;

    for token_id in expected.iter().take(mismatch_idx) {
        let token_index = usize::try_from(*token_id)?;
        let mut embed = model
            .language_model()
            .token_embedding_for_id(token_index)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        if embed.dtype() != model.dtype() {
            embed = embed.to_dtype(model.dtype())?;
        }
        model.forward(
            None,
            Some(&embed),
            None,
            None,
            None,
            None,
            None,
            Some(&mut cache_inc),
            use_cache,
        )?;
    }

    let out_inc = model.forward(
        Some(&input_ids_full),
        None,
        None,
        None,
        Some(&mask_tensor_full),
        None,
        Some(image_embeddings_slice),
        Some(&mut cache_inc),
        use_cache,
    )?;

    let full_len = cache_full.seq_len().unwrap_or(0);
    let inc_len = cache_inc.seq_len().unwrap_or(0);
    ensure!(
        full_len == prefix_len,
        "full cache seq_len {:?} != prefix_len {prefix_len}",
        cache_full.seq_len()
    );
    ensure!(
        inc_len == prefix_len,
        "inc cache seq_len {:?} != prefix_len {}",
        cache_inc.seq_len(),
        prefix_len
    );

    let hidden_full = out_full_cache.hidden_states;
    let hidden_inc = out_inc.hidden_states;
    let hidden_full_last = hidden_full
        .narrow(D::Minus2, prefix_len - 1, 1)?
        .contiguous()?;
    let hidden_inc_last = hidden_inc.narrow(D::Minus2, 0, 1)?.contiguous()?;
    let hidden_diff = max_abs_diff(&hidden_full_last, &hidden_inc_last)?;
    println!("hidden last-token max_abs diff: {hidden_diff}");

    let num_layers = model.language_model().transformer_weights().layers.len();
    for layer in 0..num_layers {
        let full_entry = cache_full
            .get(layer)
            .with_context(|| format!("missing full cache entry for layer {layer}"))?;
        let inc_entry = cache_inc
            .get(layer)
            .with_context(|| format!("missing inc cache entry for layer {layer}"))?;
        let full_k = full_entry.key_view()?;
        let inc_k = inc_entry.key_view()?;
        let full_v = full_entry.value_view()?;
        let inc_v = inc_entry.value_view()?;

        let full_last = prefix_len - 1;
        let inc_last = prefix_len - 1;
        let k_full_last = full_k.narrow(D::Minus1, full_last, 1)?.contiguous()?;
        let k_inc_last = inc_k.narrow(D::Minus1, inc_last, 1)?.contiguous()?;
        let v_full_last = full_v.narrow(D::Minus2, full_last, 1)?.contiguous()?;
        let v_inc_last = inc_v.narrow(D::Minus2, inc_last, 1)?.contiguous()?;

        let k_diff = max_abs_diff(&k_full_last, &k_inc_last)?;
        let v_diff = max_abs_diff(&v_full_last, &v_inc_last)?;
        println!("layer {layer}: key_last max_abs {k_diff} value_last max_abs {v_diff}");
    }

    Ok(())
}
