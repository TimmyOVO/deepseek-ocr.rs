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
    #[serde(rename = "input_ids")]
    _input_ids: Vec<i64>,
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

#[test]
#[ignore]
fn metal_f16_f32_logits_diff() -> Result<()> {
    let case_name = std::env::var("DEEPSEEK_OCR_METAL_DIFF_CASE")
        .unwrap_or_else(|_| "ocr1__test3_png__describe__8192_ng20".to_string());
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
        .unwrap_or(10);
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

    let mut mask = prompt.images_seq_mask.clone();
    if mask.len() < prefix_len {
        mask.extend(std::iter::repeat_n(0u8, prefix_len - mask.len()));
    } else {
        mask.truncate(prefix_len);
    }

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

    let input_ids =
        Tensor::from_vec(prefix_tokens, (1, prefix_len), &device)?.to_dtype(DType::I64)?;
    let mask_tensor = Tensor::from_vec(mask, (1, prefix_len), &device)?.to_dtype(DType::U8)?;

    let vision_f16 =
        model_f16.prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)?;
    let vision_f32 =
        model_f32.prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)?;

    let emb_f16 = model_f16.compute_image_embeddings(&[Some(vision_f16.as_ref())])?;
    let emb_f32 = model_f32.compute_image_embeddings(&[Some(vision_f32.as_ref())])?;
    let emb_f32_to_f16 = emb_f32
        .first()
        .context("missing f32 embeddings")?
        .to_dtype(DType::F16)?;
    let emb_f32_to_f16 = vec![emb_f32_to_f16];

    let out_f16 = model_f16.forward(
        Some(&input_ids),
        None,
        None,
        None,
        Some(&mask_tensor),
        None,
        Some(emb_f16.as_slice()),
        None,
        false,
    )?;
    let out_f16_f32vision = model_f16.forward(
        Some(&input_ids),
        None,
        None,
        None,
        Some(&mask_tensor),
        None,
        Some(emb_f32_to_f16.as_slice()),
        None,
        false,
    )?;
    let out_f16_f32vision_f32embeds = model_f16.forward(
        Some(&input_ids),
        None,
        None,
        None,
        Some(&mask_tensor),
        None,
        Some(emb_f32.as_slice()),
        None,
        false,
    )?;
    let out_f32 = model_f32.forward(
        Some(&input_ids),
        None,
        None,
        None,
        Some(&mask_tensor),
        None,
        Some(emb_f32.as_slice()),
        None,
        false,
    )?;

    let last_pos = prefix_len - 1;
    let logits_f16 = out_f16
        .logits
        .get(0)?
        .get(last_pos)
        .context("f16 logits missing timestep")?;
    let logits_f16_f32vision = out_f16_f32vision
        .logits
        .get(0)?
        .get(last_pos)
        .context("f16+f32vision logits missing timestep")?;
    let logits_f16_f32vision_f32embeds =
        out_f16_f32vision_f32embeds
            .logits
            .get(0)?
            .get(last_pos)
            .context("f16+f32vision(f32 embeds) logits missing timestep")?;
    let logits_f32 = out_f32
        .logits
        .get(0)?
        .get(last_pos)
        .context("f32 logits missing timestep")?;

    let (t1, v1, t2, v2) = top2_logits(&logits_f16)?;
    println!(
        "f16 native: top1 {t1} {v1} top2 {t2} {v2} margin {}",
        v1 - v2
    );
    let (t1, v1, t2, v2) = top2_logits(&logits_f16_f32vision)?;
    println!(
        "f16 + f32 vision: top1 {t1} {v1} top2 {t2} {v2} margin {}",
        v1 - v2
    );
    let (t1, v1, t2, v2) = top2_logits(&logits_f16_f32vision_f32embeds)?;
    println!(
        "f16 + f32 vision (f32 embeds): top1 {t1} {v1} top2 {t2} {v2} margin {}",
        v1 - v2
    );
    let (t1, v1, t2, v2) = top2_logits(&logits_f32)?;
    println!("f32: top1 {t1} {v1} top2 {t2} {v2} margin {}", v1 - v2);

    let hidden_f16 = out_f16.hidden_states.get(0)?.get(last_pos)?;
    let hidden_f16_f32vision = out_f16_f32vision.hidden_states.get(0)?.get(last_pos)?;
    let hidden_f16_f32vision_f32embeds = out_f16_f32vision_f32embeds
        .hidden_states
        .get(0)?
        .get(last_pos)?;
    let hidden_f32 = out_f32.hidden_states.get(0)?.get(last_pos)?;

    let h16 = hidden_f16.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let h16_f32vision = hidden_f16_f32vision
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let h16_f32vision_f32embeds = hidden_f16_f32vision_f32embeds
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let h32 = hidden_f32.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    let max_abs_f16 = h16
        .iter()
        .zip(h32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_abs_f16_f32vision = h16_f32vision
        .iter()
        .zip(h32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_abs_f16_f32vision_f32embeds = h16_f32vision_f32embeds
        .iter()
        .zip(h32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("hidden max_abs f16 vs f32: {max_abs_f16}");
    println!("hidden max_abs f16+f32vision vs f32: {max_abs_f16_f32vision}");
    println!("hidden max_abs f16+f32vision(f32 embeds) vs f32: {max_abs_f16_f32vision_f32embeds}");

    Ok(())
}
