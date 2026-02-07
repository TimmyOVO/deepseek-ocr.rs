mod common;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use deepseek_ocr_infer_deepseek::{
    config::load_ocr_config,
    model::DeepseekOcrModel,
    transformer::{block::TransformerBlock, block::build_attention_bias, rope::RopeCache},
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

#[test]
#[ignore]
fn metal_f16_f32_layer_diff() -> Result<()> {
    // Fixed failing case in OCR1 metal f16 gate (override via env).
    let case_name = std::env::var("DEEPSEEK_OCR_METAL_DIFF_CASE")
        .unwrap_or_else(|_| "ocr1__test3_png__grounding_md__8192_ng20".to_string());
    let case_dir = workspace_path(format!("baselines/long/{case_name}"));
    let baseline_path = case_dir.join("baseline.json");
    let baseline: BaselineMetadata = serde_json::from_str(&fs::read_to_string(&baseline_path)?)?;

    let prompt_override = std::env::var("DEEPSEEK_OCR_METAL_DIFF_PROMPT_ASSETS")
        .ok()
        .map(workspace_path);
    let prompt_path = prompt_override.unwrap_or_else(|| {
        baseline
            .prompt_assets_path
            .as_deref()
            .map(workspace_path)
            .unwrap_or_else(|| case_dir.join("prompt.json"))
    });
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
        .unwrap_or(328);
    let prefill_len = output.prefill_len;
    let mut expected = output.tokens[prefill_len..].to_vec();
    if let Some(eos) = output.eos_token_id
        && expected.last().copied() == Some(eos)
    {
        expected.pop();
    }
    anyhow::ensure!(
        mismatch_idx < expected.len(),
        "mismatch index {} exceeds expected length {}",
        mismatch_idx,
        expected.len()
    );

    let prefix_tokens = if std::env::var("DEEPSEEK_OCR_METAL_DIFF_USE_BASELINE_PREFIX")
        .ok()
        .as_deref()
        == Some("1")
    {
        output.tokens[..prefill_len + mismatch_idx].to_vec()
    } else {
        let mut tokens = Vec::with_capacity(prompt.input_ids.len() + mismatch_idx);
        tokens.extend_from_slice(&prompt.input_ids);
        tokens.extend_from_slice(&expected[..mismatch_idx]);
        tokens
    };
    let seq_len = prefix_tokens.len();

    let mut mask = prompt.images_seq_mask.clone();
    if mask.len() < seq_len {
        mask.extend(std::iter::repeat_n(0u8, seq_len - mask.len()));
    } else {
        mask.truncate(seq_len);
    }

    let attention_vec = vec![1i64; seq_len];

    let image_path = workspace_path(&baseline.image);
    let image = image::open(&image_path)
        .with_context(|| format!("failed to open image at {}", image_path.display()))?;

    let base_size = baseline.base_size.unwrap_or(1024);
    let image_size = baseline.image_size.unwrap_or(640);
    let crop_mode = baseline.crop_mode.unwrap_or(true);

    let model_dir = baseline.model.as_deref().unwrap_or("DeepSeek-OCR");
    let config_path = workspace_path(format!("{model_dir}/config.json"));
    let weights_path = workspace_path(format!("{model_dir}/model-00001-of-000001.safetensors"));

    let device_f16 = Device::new_metal(0).context("failed to init metal device")?;
    let device_f32 = device_f16.clone();

    let cfg = load_ocr_config(Some(&config_path))?;
    let language_cfg = cfg.resolved_language_config()?;

    let model_f16 = DeepseekOcrModel::load(
        Some(&config_path),
        Some(&weights_path),
        None,
        device_f16.clone(),
        DType::F16,
    )?;
    let model_f32 = DeepseekOcrModel::load(
        Some(&config_path),
        Some(&weights_path),
        None,
        device_f32.clone(),
        DType::F32,
    )?;

    let use_f32_cache = std::env::var("DEEPSEEK_OCR_METAL_DIFF_F32_CACHE")
        .ok()
        .as_deref()
        == Some("1");

    let input_ids_f16 =
        Tensor::from_vec(prefix_tokens.clone(), (1, seq_len), &device_f16)?.to_dtype(DType::I64)?;
    let input_ids_f32 =
        Tensor::from_vec(prefix_tokens.clone(), (1, seq_len), &device_f32)?.to_dtype(DType::I64)?;

    let mask_f16 =
        Tensor::from_vec(mask.clone(), (1, seq_len), &device_f16)?.to_dtype(DType::U8)?;
    let mask_f32 =
        Tensor::from_vec(mask.clone(), (1, seq_len), &device_f32)?.to_dtype(DType::U8)?;

    let attention_f16 =
        Tensor::from_vec(attention_vec.clone(), (1, seq_len), &device_f16)?.to_dtype(DType::I64)?;
    let attention_f32 =
        Tensor::from_vec(attention_vec.clone(), (1, seq_len), &device_f32)?.to_dtype(DType::I64)?;

    let vision_f32 =
        model_f32.prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)?;

    let embeddings_f32 = model_f32.compute_image_embeddings(&[Some(vision_f32.as_ref())])?;

    let token_embeddings_f16 = model_f16.language_model().embed_tokens(&input_ids_f16)?;
    let token_embeddings_f32 = model_f32.language_model().embed_tokens(&input_ids_f32)?;

    let injected_f16 = model_f16.inject_image_tokens_for_tests(
        token_embeddings_f16.clone(),
        &mask_f16,
        Some(embeddings_f32.as_slice()),
    )?;
    let injected_f32 = model_f32.inject_image_tokens_for_tests(
        token_embeddings_f32.clone(),
        &mask_f32,
        Some(embeddings_f32.as_slice()),
    )?;

    let rope_dim_cfg = language_cfg
        .qk_rope_head_dim
        .unwrap_or(language_cfg.hidden_size / language_cfg.num_attention_heads);
    let rope_dim = if rope_dim_cfg == 0 {
        language_cfg.hidden_size / language_cfg.num_attention_heads
    } else {
        rope_dim_cfg
    };

    let last_pos = seq_len - 1;
    let embed_last_f16 = injected_f16.get(0)?.get(last_pos)?.to_dtype(DType::F32)?;
    let embed_last_f32 = injected_f32.get(0)?.get(last_pos)?.to_dtype(DType::F32)?;
    let embed_vec_f16 = embed_last_f16.to_vec1::<f32>()?;
    let embed_vec_f32 = embed_last_f32.to_vec1::<f32>()?;
    let embed_max = max_abs_diff(&embed_vec_f16, &embed_vec_f32);

    let token_embed_max = max_abs_diff(
        &vec_f32(&token_embeddings_f16)?,
        &vec_f32(&token_embeddings_f32)?,
    );
    let embed_all_max = max_abs_diff(&vec_f32(&injected_f16)?, &vec_f32(&injected_f32)?);

    let mut rope_f16 = RopeCache::new(&device_f16, DType::F16, rope_dim)?;
    rope_f16.ensure_len(&language_cfg, seq_len)?;
    let rope_tensors_f16 = rope_f16.select(1, seq_len, None)?;

    let mut rope_f32 = RopeCache::new(&device_f32, DType::F32, rope_dim)?;
    rope_f32.ensure_len(&language_cfg, seq_len)?;
    let rope_tensors_f32 = rope_f32.select(1, seq_len, None)?;

    let bias_dtype_f16 = DType::F16;
    let bias_dtype_f32 = DType::F32;
    let attn_bias_f16 = build_attention_bias(
        Some(&attention_f16),
        1,
        seq_len,
        seq_len,
        0,
        bias_dtype_f16,
        &device_f16,
    )?;
    let attn_bias_f32 = build_attention_bias(
        Some(&attention_f32),
        1,
        seq_len,
        seq_len,
        0,
        bias_dtype_f32,
        &device_f32,
    )?;

    if use_f32_cache {
        let mut cache_f16 = model_f16.new_cache_for_dtype(DType::F16)?;
        cache_f16.clear();
        let _ = model_f16.forward(
            Some(&input_ids_f16),
            None,
            None,
            None,
            Some(&mask_f16),
            None,
            Some(embeddings_f32.as_slice()),
            Some(&mut cache_f16),
            true,
        )?;

        let mut cache_f32 = model_f32.new_cache_for_dtype(DType::F32)?;
        cache_f32.clear();
        let _ = model_f32.forward(
            Some(&input_ids_f32),
            None,
            None,
            None,
            Some(&mask_f32),
            None,
            Some(embeddings_f32.as_slice()),
            Some(&mut cache_f32),
            true,
        )?;

        for layer in 0..cache_f16.num_layers() {
            let f16_entry = cache_f16.get(layer).context("missing f16 cache")?;
            let f32_entry = cache_f32.get(layer).context("missing f32 cache")?;
            let key_max = max_abs_diff(
                &vec_f32(&f16_entry.key_view()?)?,
                &vec_f32(&f32_entry.key_view()?)?,
            );
            let val_max = max_abs_diff(
                &vec_f32(&f16_entry.value_view()?)?,
                &vec_f32(&f32_entry.value_view()?)?,
            );
            println!("prefill cache layer {layer}: key_max_abs {key_max} value_max_abs {val_max}");
        }
    }

    let weights_f16 = model_f16.language_model().transformer_weights();
    let weights_f32 = model_f32.language_model().transformer_weights();

    let mut hidden_f16 = injected_f16;
    let mut hidden_f32 = injected_f32;

    let mut layer_diffs = Vec::new();
    for (idx, layer_weights_f16) in weights_f16.layers.iter().enumerate() {
        let block_f16 = TransformerBlock::new(&language_cfg, layer_weights_f16, false);
        let block_f32 = TransformerBlock::new(&language_cfg, &weights_f32.layers[idx], false);

        let out_f16 = block_f16.forward(
            idx,
            &hidden_f16,
            attn_bias_f16.as_ref(),
            Some((&rope_tensors_f16.0, &rope_tensors_f16.1)),
            None,
            false,
        )?;
        let out_f32 = block_f32.forward(
            idx,
            &hidden_f32,
            attn_bias_f32.as_ref(),
            Some((&rope_tensors_f32.0, &rope_tensors_f32.1)),
            None,
            false,
        )?;

        hidden_f16 = out_f16.hidden_states;
        hidden_f32 = out_f32.hidden_states;

        let last_f16 = hidden_f16.get(0)?.get(last_pos)?.to_dtype(DType::F32)?;
        let last_f32 = hidden_f32.get(0)?.get(last_pos)?.to_dtype(DType::F32)?;
        let v16 = last_f16.to_vec1::<f32>()?;
        let v32 = last_f32.to_vec1::<f32>()?;
        let max = max_abs_diff(&v16, &v32);
        layer_diffs.push((idx, max));
    }

    let out_dir = workspace_path("session/tmp_case_test3_step328");
    fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join("layer_diff.json");
    #[derive(serde::Serialize)]
    struct Report {
        case_dir: String,
        seq_len: usize,
        mismatch_idx: usize,
        embed_last_max_abs: f32,
        embed_all_max_abs: f32,
        token_embed_max_abs: f32,
        layers: Vec<(usize, f32)>,
    }
    let report = Report {
        case_dir: case_dir.display().to_string(),
        seq_len,
        mismatch_idx,
        embed_last_max_abs: embed_max,
        embed_all_max_abs: embed_all_max,
        token_embed_max_abs: token_embed_max,
        layers: layer_diffs,
    };
    fs::write(out_path, serde_json::to_vec_pretty(&report)?)?;
    println!("embed_last_max_abs: {embed_max}");
    println!("embed_all_max_abs: {embed_all_max}");
    println!("token_embed_max_abs: {token_embed_max}");
    for (idx, max) in report.layers.iter() {
        println!("layer {idx}: max_abs {max}");
    }
    Ok(())
}
