mod common;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use deepseek_ocr_infer_deepseek::{config::load_ocr_config, model::DeepseekOcrModel};
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

fn top2(values: &[f32]) -> Option<(usize, f32, usize, f32)> {
    if values.is_empty() {
        return None;
    }
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut second_i = 0usize;
    let mut second_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        if v > best_v {
            second_i = best_i;
            second_v = best_v;
            best_i = i;
            best_v = v;
        } else if v > second_v {
            second_i = i;
            second_v = v;
        }
    }
    Some((best_i, best_v, second_i, second_v))
}

fn top2_logits(logits_1d: &Tensor) -> Result<(usize, f32, usize, f32)> {
    let vec = logits_1d
        .to_dtype(DType::F32)?
        .contiguous()?
        .to_vec1::<f32>()?;
    top2(&vec).context("empty logits")
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
fn metal_f16_decode_cache_f32() -> Result<()> {
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
        && expected.last().copied() == Some(eos) {
        expected.pop();
    }
    ensure!(
        mismatch_idx < expected.len(),
        "mismatch index {} exceeds expected length {}",
        mismatch_idx,
        expected.len()
    );
    ensure!(mismatch_idx > 0, "mismatch_idx must be > 0");

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
    let _cfg = load_ocr_config(Some(&config_path))?;

    let model = DeepseekOcrModel::load(
        Some(&config_path),
        Some(&weights_path),
        None,
        device.clone(),
        DType::F16,
    )?;

    let vision = model.prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)?;
    let image_embeddings = model.compute_image_embeddings(&[Some(vision.as_ref())])?;

    let prefill_ids = Tensor::from_vec(prompt.input_ids.clone(), (1, prefill_len), &device)?
        .to_dtype(DType::I64)?;
    let prefill_mask = Tensor::from_vec(prompt.images_seq_mask.clone(), (1, prefill_len), &device)?
        .to_dtype(DType::U8)?;

    let mut cache_f16 = model.new_cache_for_dtype(DType::F16)?;
    cache_f16.clear();
    model.forward(
        Some(&prefill_ids),
        None,
        None,
        None,
        Some(&prefill_mask),
        None,
        Some(image_embeddings.as_slice()),
        Some(&mut cache_f16),
        true,
    )?;

    let mut cache_f32 = model.new_cache_for_dtype(DType::F16)?;
    cache_f32.clear();
    model.forward(
        Some(&prefill_ids),
        None,
        None,
        None,
        Some(&prefill_mask),
        None,
        Some(image_embeddings.as_slice()),
        Some(&mut cache_f32),
        true,
    )?;

    for token_id in expected.iter().take(mismatch_idx - 1) {
        decode_step(&model, &mut cache_f16, *token_id)?;
        decode_step(&model, &mut cache_f32, *token_id)?;
    }

    let target_token = expected[mismatch_idx - 1];
    let mut embed = model
        .language_model()
        .token_embedding_for_id(usize::try_from(target_token)?)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    if embed.dtype() != model.dtype() {
        embed = embed.to_dtype(model.dtype())?;
    }

    let mut cache_f16_step = cache_f16.clone();
    let out_f16 = model.forward(
        None,
        Some(&embed),
        None,
        None,
        None,
        None,
        None,
        Some(&mut cache_f16_step),
        true,
    )?;

    let mut cache_f32_step = cache_f32.clone();
    let out_f32 = model.forward(
        None,
        Some(&embed),
        None,
        None,
        None,
        None,
        None,
        Some(&mut cache_f32_step),
        true,
    )?;

    let logits_f16 = out_f16.logits.get(0)?.get(0)?;
    let logits_f32 = out_f32.logits.get(0)?.get(0)?;

    let (t1, v1, t2, v2) = top2_logits(&logits_f16)?;
    println!(
        "f16 cache: top1 {t1} {v1} top2 {t2} {v2} margin {}",
        v1 - v2
    );
    let (t1, v1, t2, v2) = top2_logits(&logits_f32)?;
    println!(
        "f16 + f32 cache: top1 {t1} {v1} top2 {t2} {v2} margin {}",
        v1 - v2
    );

    Ok(())
}
