mod common;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use deepseek_ocr_infer_deepseek::{
    config::load_ocr_config,
    model::DeepseekOcrModel,
    transformer::{block::TransformerBlock, rope::RopeCache},
};
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn vec_f32(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?)
}

fn avg_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += (x - y).abs() as f64;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        (sum / (n as f64)) as f32
    }
}

fn decode_step(
    model: &DeepseekOcrModel,
    cache: &mut deepseek_ocr_core::cache::DynamicCache,
    token_id: i64,
) -> Result<()> {
    let token_index = usize::try_from(token_id)?;
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
        Some(cache),
        true,
    )?;
    Ok(())
}

#[test]
#[ignore]
fn metal_f16_f32_cache_layer_diff() -> Result<()> {
    let case_name = std::env::var("DEEPSEEK_OCR_METAL_DIFF_CASE")
        .unwrap_or_else(|_| "ocr1__test_png__grounding_md__8192_ng20".to_string());
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
        .unwrap_or(455);
    let prefill_len = output.prefill_len;
    let mut expected = output.tokens[prefill_len..].to_vec();
    if let Some(eos) = output.eos_token_id
        && expected.last().copied() == Some(eos)
    {
        expected.pop();
    }
    ensure!(
        mismatch_idx < expected.len(),
        "mismatch index {} exceeds expected length {}",
        mismatch_idx,
        expected.len()
    );
    ensure!(mismatch_idx > 0, "mismatch_idx must be > 0 for cache diff");

    let image_path = workspace_path(&baseline.image);
    let image = image::open(&image_path)
        .with_context(|| format!("failed to open image at {}", image_path.display()))?;

    let base_size = baseline.base_size.unwrap_or(1024);
    let image_size = baseline.image_size.unwrap_or(640);
    let crop_mode = baseline.crop_mode.unwrap_or(true);

    let model_dir = baseline.model.as_deref().unwrap_or("DeepSeek-OCR");
    let config_path = workspace_path(format!("{model_dir}/config.json"));
    let weights_path = workspace_path(format!("{model_dir}/model-00001-of-000001.safetensors"));

    let device = Device::new_metal(0).context("failed to init metal device")?;
    let cfg = load_ocr_config(Some(&config_path))?;
    let language_cfg = cfg.resolved_language_config()?;

    let model_f16 = DeepseekOcrModel::load(
        Some(&config_path),
        Some(&weights_path),
        None,
        device.clone(),
        DType::F16,
    )?;
    let model_f32 = DeepseekOcrModel::load(
        Some(&config_path),
        Some(&weights_path),
        None,
        device.clone(),
        DType::F32,
    )?;

    let vision_f16 =
        model_f16.prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)?;
    let vision_f32 =
        model_f32.prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)?;

    let emb_f16 = model_f16.compute_image_embeddings(&[Some(vision_f16.as_ref())])?;
    let emb_f32 = model_f32.compute_image_embeddings(&[Some(vision_f32.as_ref())])?;

    let prefill_ids = Tensor::from_vec(prompt.input_ids.clone(), (1, prefill_len), &device)?
        .to_dtype(DType::I64)?;
    let prefill_mask = Tensor::from_vec(prompt.images_seq_mask.clone(), (1, prefill_len), &device)?
        .to_dtype(DType::U8)?;

    let mut cache_f16 = model_f16.new_cache_for_dtype(DType::F16)?;
    cache_f16.clear();
    model_f16.forward(
        Some(&prefill_ids),
        None,
        None,
        None,
        Some(&prefill_mask),
        None,
        Some(emb_f16.as_slice()),
        Some(&mut cache_f16),
        true,
    )?;

    let mut cache_f32 = model_f32.new_cache_for_dtype(DType::F32)?;
    cache_f32.clear();
    model_f32.forward(
        Some(&prefill_ids),
        None,
        None,
        None,
        Some(&prefill_mask),
        None,
        Some(emb_f32.as_slice()),
        Some(&mut cache_f32),
        true,
    )?;

    for token_id in expected.iter().take(mismatch_idx - 1) {
        decode_step(&model_f16, &mut cache_f16, *token_id)?;
        decode_step(&model_f32, &mut cache_f32, *token_id)?;
    }

    let past_len = cache_f16.seq_len().unwrap_or(0);
    let past_len_f32 = cache_f32.seq_len().unwrap_or(0);
    ensure!(
        past_len == past_len_f32,
        "cache len mismatch f16={} f32={}",
        past_len,
        past_len_f32
    );

    let target_token = expected[mismatch_idx - 1];
    let mut embed_f16 = model_f16
        .language_model()
        .token_embedding_for_id(usize::try_from(target_token)?)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    if embed_f16.dtype() != model_f16.dtype() {
        embed_f16 = embed_f16.to_dtype(model_f16.dtype())?;
    }
    let mut embed_f32 = model_f32
        .language_model()
        .token_embedding_for_id(usize::try_from(target_token)?)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    if embed_f32.dtype() != model_f32.dtype() {
        embed_f32 = embed_f32.to_dtype(model_f32.dtype())?;
    }

    let rope_dim_cfg = language_cfg
        .qk_rope_head_dim
        .unwrap_or(language_cfg.hidden_size / language_cfg.num_attention_heads);
    let rope_dim = if rope_dim_cfg == 0 {
        language_cfg.hidden_size / language_cfg.num_attention_heads
    } else {
        rope_dim_cfg
    };

    let mut rope_f16 = RopeCache::new(&device, DType::F16, rope_dim)?;
    rope_f16.ensure_len(&language_cfg, past_len + 1)?;
    let ids_f16 = Tensor::from_vec(vec![past_len as i64], (1, 1), &device)?.to_dtype(DType::I64)?;
    let rope_tensors_f16 = rope_f16.select(1, 1, Some(&ids_f16))?;

    let mut rope_f32 = RopeCache::new(&device, DType::F32, rope_dim)?;
    rope_f32.ensure_len(&language_cfg, past_len + 1)?;
    let ids_f32 = Tensor::from_vec(vec![past_len as i64], (1, 1), &device)?.to_dtype(DType::I64)?;
    let rope_tensors_f32 = rope_f32.select(1, 1, Some(&ids_f32))?;

    for layer in 0..model_f16
        .language_model()
        .transformer_weights()
        .layers
        .len()
    {
        let entry_f16 = cache_f16.get(layer).context("missing f16 cache entry")?;
        let entry_f32 = cache_f32.get(layer).context("missing f32 cache entry")?;
        let key_f16 = entry_f16.key_view()?;
        let key_f32 = entry_f32.key_view()?;
        let val_f16 = entry_f16.value_view()?;
        let val_f32 = entry_f32.value_view()?;
        let key_f16_v = vec_f32(&key_f16)?;
        let key_f32_v = vec_f32(&key_f32)?;
        let val_f16_v = vec_f32(&val_f16)?;
        let val_f32_v = vec_f32(&val_f32)?;
        let key_max = max_abs_diff(&key_f16_v, &key_f32_v);
        let val_max = max_abs_diff(&val_f16_v, &val_f32_v);
        let key_avg = avg_abs_diff(&key_f16_v, &key_f32_v);
        let val_avg = avg_abs_diff(&val_f16_v, &val_f32_v);
        println!(
            "cache layer {layer}: key_max_abs {key_max} key_avg_abs {key_avg} value_max_abs {val_max} value_avg_abs {val_avg}"
        );
    }

    let weights_f16 = model_f16.language_model().transformer_weights();
    let weights_f32 = model_f32.language_model().transformer_weights();

    let mut hidden_f16 = embed_f16;
    let mut hidden_f32 = embed_f32;
    for (idx, layer_weights_f16) in weights_f16.layers.iter().enumerate() {
        let block_f16 = TransformerBlock::new(&language_cfg, layer_weights_f16, false);
        let block_f32 = TransformerBlock::new(&language_cfg, &weights_f32.layers[idx], false);
        let out_f16 = block_f16.forward(
            idx,
            &hidden_f16,
            None,
            Some((&rope_tensors_f16.0, &rope_tensors_f16.1)),
            cache_f16.get(idx),
            false,
        )?;
        let out_f32 = block_f32.forward(
            idx,
            &hidden_f32,
            None,
            Some((&rope_tensors_f32.0, &rope_tensors_f32.1)),
            cache_f32.get(idx),
            false,
        )?;

        let v16 = vec_f32(&out_f16.hidden_states)?;
        let v32 = vec_f32(&out_f32.hidden_states)?;
        let max = max_abs_diff(&v16, &v32);
        println!("layer {idx}: max_abs {max}");

        hidden_f16 = out_f16.hidden_states;
        hidden_f32 = out_f32.hidden_states;
    }

    Ok(())
}
