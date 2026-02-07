mod common;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use deepseek_ocr_infer_deepseek::model::DeepseekOcrModel;
use serde::Deserialize;
use std::fs;

use common::test_utils::workspace_path;

#[derive(Debug, Deserialize)]
struct BaselineMetadata {
    image: String,
    #[serde(default)]
    base_size: Option<u32>,
    #[serde(default)]
    image_size: Option<u32>,
    #[serde(default)]
    crop_mode: Option<bool>,
}

fn max_abs_diff_tensor_any(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?;
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

fn print_diff(label: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    let diff = max_abs_diff_tensor_any(a, b)?;
    println!("{label}: {diff}");
    Ok(())
}

#[test]
#[ignore]
fn metal_f16_f32_vision_diff() -> Result<()> {
    let case_name = std::env::var("DEEPSEEK_OCR_METAL_DIFF_CASE")
        .unwrap_or_else(|_| "ocr1__test3_png__describe__8192_ng20".to_string());
    let case_dir = workspace_path(format!("baselines/long/{case_name}"));
    let baseline_path = case_dir.join("baseline.json");
    let baseline: BaselineMetadata = serde_json::from_str(&fs::read_to_string(&baseline_path)?)?;

    let image_path = workspace_path(&baseline.image);
    let image = image::open(&image_path)
        .with_context(|| format!("failed to open image at {}", image_path.display()))?;

    let base_size = baseline.base_size.unwrap_or(1024);
    let image_size = baseline.image_size.unwrap_or(640);
    let crop_mode = baseline.crop_mode.unwrap_or(true);

    let config_path = workspace_path("DeepSeek-OCR/config.json");
    let weights_path = workspace_path("DeepSeek-OCR/model-00001-of-000001.safetensors");

    let device = Device::new_metal(0).context("failed to init metal device")?;
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

    let input_f16 = model_f16
        .prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)
        .context("prepare vision input f16")?;
    let input_f32 = model_f32
        .prepare_vision_input_from_image(&image, base_size, image_size, crop_mode)
        .context("prepare vision input f32")?;

    let debug_f16 = model_f16
        .compute_vision_debug_features(&input_f16.as_ref())
        .context("vision debug f16")?;
    let debug_f32 = model_f32
        .compute_vision_debug_features(&input_f32.as_ref())
        .context("vision debug f32")?;

    println!("== SAM global ==");
    print_diff(
        "sam.global.patch_embed",
        &debug_f16.global_sam_trace.patch_embed,
        &debug_f32.global_sam_trace.patch_embed,
    )?;
    if let (Some(pos_f16), Some(pos_f32)) = (
        debug_f16.global_sam_trace.pos_added.as_ref(),
        debug_f32.global_sam_trace.pos_added.as_ref(),
    ) {
        print_diff("sam.global.pos_added", pos_f16, pos_f32)?;
    }
    for (idx, (t16, t32)) in debug_f16
        .global_sam_trace
        .block_outputs
        .iter()
        .zip(debug_f32.global_sam_trace.block_outputs.iter())
        .enumerate()
    {
        print_diff(&format!("sam.global.block[{idx}]"), t16, t32)?;
    }
    print_diff(
        "sam.global.neck_conv1",
        &debug_f16.global_sam_trace.neck_conv1,
        &debug_f32.global_sam_trace.neck_conv1,
    )?;
    print_diff(
        "sam.global.neck_norm1",
        &debug_f16.global_sam_trace.neck_norm1,
        &debug_f32.global_sam_trace.neck_norm1,
    )?;
    print_diff(
        "sam.global.neck_conv2",
        &debug_f16.global_sam_trace.neck_conv2,
        &debug_f32.global_sam_trace.neck_conv2,
    )?;
    print_diff(
        "sam.global.neck_norm2",
        &debug_f16.global_sam_trace.neck_norm2,
        &debug_f32.global_sam_trace.neck_norm2,
    )?;
    print_diff(
        "sam.global.net2",
        &debug_f16.global_sam_trace.net2,
        &debug_f32.global_sam_trace.net2,
    )?;
    print_diff(
        "sam.global.net3",
        &debug_f16.global_sam_trace.net3,
        &debug_f32.global_sam_trace.net3,
    )?;

    if let (Some(local_sam_f16), Some(local_sam_f32)) = (
        debug_f16.local_sam_trace.as_ref(),
        debug_f32.local_sam_trace.as_ref(),
    ) {
        println!("== SAM local ==");
        print_diff(
            "sam.local.patch_embed",
            &local_sam_f16.patch_embed,
            &local_sam_f32.patch_embed,
        )?;
        if let (Some(pos_f16), Some(pos_f32)) = (
            local_sam_f16.pos_added.as_ref(),
            local_sam_f32.pos_added.as_ref(),
        ) {
            print_diff("sam.local.pos_added", pos_f16, pos_f32)?;
        }
        for (idx, (t16, t32)) in local_sam_f16
            .block_outputs
            .iter()
            .zip(local_sam_f32.block_outputs.iter())
            .enumerate()
        {
            print_diff(&format!("sam.local.block[{idx}]"), t16, t32)?;
        }
        print_diff(
            "sam.local.neck_conv1",
            &local_sam_f16.neck_conv1,
            &local_sam_f32.neck_conv1,
        )?;
        print_diff(
            "sam.local.neck_norm1",
            &local_sam_f16.neck_norm1,
            &local_sam_f32.neck_norm1,
        )?;
        print_diff(
            "sam.local.neck_conv2",
            &local_sam_f16.neck_conv2,
            &local_sam_f32.neck_conv2,
        )?;
        print_diff(
            "sam.local.neck_norm2",
            &local_sam_f16.neck_norm2,
            &local_sam_f32.neck_norm2,
        )?;
        print_diff("sam.local.net2", &local_sam_f16.net2, &local_sam_f32.net2)?;
        print_diff("sam.local.net3", &local_sam_f16.net3, &local_sam_f32.net3)?;
    }

    println!("== CLIP global ==");
    print_diff(
        "clip.global.embeddings",
        &debug_f16.global_clip_trace.embeddings,
        &debug_f32.global_clip_trace.embeddings,
    )?;
    print_diff(
        "clip.global.pre_layernorm",
        &debug_f16.global_clip_trace.pre_layernorm,
        &debug_f32.global_clip_trace.pre_layernorm,
    )?;
    for (idx, (t16, t32)) in debug_f16
        .global_clip_trace
        .layer_outputs
        .iter()
        .zip(debug_f32.global_clip_trace.layer_outputs.iter())
        .enumerate()
    {
        print_diff(&format!("clip.global.block[{idx}]"), t16, t32)?;
    }

    if let (Some(local_clip_f16), Some(local_clip_f32)) = (
        debug_f16.local_clip_trace.as_ref(),
        debug_f32.local_clip_trace.as_ref(),
    ) {
        println!("== CLIP local ==");
        print_diff(
            "clip.local.embeddings",
            &local_clip_f16.embeddings,
            &local_clip_f32.embeddings,
        )?;
        print_diff(
            "clip.local.pre_layernorm",
            &local_clip_f16.pre_layernorm,
            &local_clip_f32.pre_layernorm,
        )?;
        for (idx, (t16, t32)) in local_clip_f16
            .layer_outputs
            .iter()
            .zip(local_clip_f32.layer_outputs.iter())
            .enumerate()
        {
            print_diff(&format!("clip.local.block[{idx}]"), t16, t32)?;
        }
    }

    let proj_f16 = model_f16
        .compute_vision_projection(&input_f16.as_ref())
        .context("vision projection f16")?;
    let proj_f32 = model_f32
        .compute_vision_projection(&input_f32.as_ref())
        .context("vision projection f32")?;

    println!("== Projector ==");
    print_diff(
        "projector.global_pre",
        &proj_f16.global_pre,
        &proj_f32.global_pre,
    )?;
    if let (Some(local_pre_f16), Some(local_pre_f32)) =
        (proj_f16.local_pre.as_ref(), proj_f32.local_pre.as_ref())
    {
        print_diff("projector.local_pre", local_pre_f16, local_pre_f32)?;
    }
    print_diff(
        "projector.global_post",
        &proj_f16.global_post,
        &proj_f32.global_post,
    )?;
    if let (Some(local_post_f16), Some(local_post_f32)) =
        (proj_f16.local_post.as_ref(), proj_f32.local_post.as_ref())
    {
        print_diff("projector.local_post", local_post_f16, local_post_f32)?;
    }
    print_diff(
        "projector.global_tokens",
        &proj_f16.global_tokens,
        &proj_f32.global_tokens,
    )?;
    if let (Some(local_tokens_f16), Some(local_tokens_f32)) = (
        proj_f16.local_tokens.as_ref(),
        proj_f32.local_tokens.as_ref(),
    ) {
        print_diff("projector.local_tokens", local_tokens_f16, local_tokens_f32)?;
    }
    print_diff(
        "projector.fused_tokens",
        &proj_f16.fused_tokens,
        &proj_f32.fused_tokens,
    )?;

    Ok(())
}
