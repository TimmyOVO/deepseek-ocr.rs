use tracing::trace;

use anyhow::{Context, Result, anyhow};
use candle_core::Tensor;
use image::DynamicImage;
use tokenizers::Tokenizer;

use crate::{
    benchmark::Timer,
    conversation::get_conv_template,
    model::{DeepseekOcrModel, OwnedVisionInput, VisionInput},
};

/// Render a prompt using the configured conversation template and system prompt.
pub fn render_prompt(template: &str, system_prompt: &str, raw_prompt: &str) -> Result<String> {
    let timer = Timer::new("prompt.render");
    let mut template = get_conv_template(template)
        .with_context(|| format!("unknown conversation template {template}"))?;
    template.set_system_message(system_prompt.to_owned());
    template.reset_messages();
    template.append_message("User", Some(raw_prompt.to_owned()));
    template.append_message("Assistant", None);
    let prompt = template.get_prompt();
    timer.finish(|event| {
        event.add_field("chars", prompt.len() as u64);
    });
    Ok(prompt)
}

/// Prepare SAM/CLIP inputs for the provided images.
pub fn prepare_vision_inputs(
    model: &DeepseekOcrModel,
    images: &[DynamicImage],
    base_size: u32,
    image_size: u32,
    crop_mode: bool,
) -> Result<Vec<OwnedVisionInput>> {
    let timer = Timer::new("vision.prepare_inputs");
    if !images.is_empty() {
        trace!(
            "Preparing vision input (base_size={base_size}, image_size={image_size}, crop_mode={crop_mode})"
        );
    }
    let result = images
        .iter()
        .map(|image| {
            model
                .prepare_vision_input_from_image(image, base_size, image_size, crop_mode)
                .with_context(|| "failed to build vision input")
        })
        .collect::<Result<Vec<_>>>();
    match &result {
        Ok(inputs) => {
            timer.finish(|event| {
                event.add_field("images", inputs.len());
                event.add_field("base_size", base_size as u64);
                event.add_field("image_size", image_size as u64);
                event.add_field("crop_mode", crop_mode);
            });
        }
        Err(_) => {
            timer.finish(|_| {});
        }
    }
    result
}

/// Compute image embeddings for the prepared SAM inputs.
pub fn compute_image_embeddings(
    model: &DeepseekOcrModel,
    owned_inputs: &[OwnedVisionInput],
) -> Result<Vec<Tensor>> {
    let timer = Timer::new("vision.compute_embeddings");
    if owned_inputs.is_empty() {
        timer.finish(|event| {
            event.add_field("images", 0u64);
        });
        return Ok(Vec::new());
    }
    let refs: Vec<Option<VisionInput<'_>>> = owned_inputs
        .iter()
        .map(|owned| Some(owned.as_ref()))
        .collect();
    trace!("Computing image embeddings for {} image(s)...", refs.len());
    let outputs = model.compute_image_embeddings(&refs);
    match &outputs {
        Ok(values) => {
            let tokens_total: u64 = values
                .iter()
                .map(|tensor| tensor.shape().dims().first().copied().unwrap_or(0) as u64)
                .sum();
            timer.finish(|event| {
                event.add_field("images", refs.len());
                event.add_field("device_is_cuda", model.device().is_cuda());
                event.add_field("device_is_metal", model.device().is_metal());
                event.add_field("token_rows_total", tokens_total);
            });
        }
        Err(_) => {
            timer.finish(|_| {});
        }
    }
    outputs
}

/// Tokenise a prompt and align `<image>` placeholders with the computed embeddings.
pub fn build_prompt_tokens(
    tokenizer: &Tokenizer,
    prompt: &str,
    embeddings: &[Tensor],
    vision_inputs: &[OwnedVisionInput],
    base_size: u32,
    image_size: u32,
    crop_mode: bool,
) -> Result<(Vec<i64>, Vec<u8>)> {
    let timer = Timer::new("prompt.build_tokens");
    let image_token_id = tokenizer
        .token_to_id("<image>")
        .ok_or_else(|| anyhow!("tokenizer missing <image> token"))? as i64;
    let bos_id = 0i64;

    let segments: Vec<&str> = prompt.split("<image>").collect();
    anyhow::ensure!(
        segments.len().saturating_sub(1) == embeddings.len(),
        "prompt/image embedding mismatch: {} slots vs {} embeddings",
        segments.len().saturating_sub(1),
        embeddings.len()
    );
    anyhow::ensure!(
        embeddings.len() == vision_inputs.len(),
        "vision input count {} does not match embeddings {}",
        vision_inputs.len(),
        embeddings.len()
    );

    let mut tokens = Vec::new();
    let mut mask = Vec::new();
    tokens.push(bos_id);
    mask.push(0);

    for (idx, segment) in segments.iter().enumerate() {
        let encoding = tokenizer
            .encode(*segment, false)
            .map_err(|err| anyhow!("tokenization failed: {err}"))?;
        tokens.extend(encoding.get_ids().iter().map(|&id| id as i64));
        mask.extend(std::iter::repeat(0u8).take(encoding.len()));
        if idx < embeddings.len() {
            let placeholders = build_image_placeholders(
                image_token_id,
                &vision_inputs[idx],
                embeddings[idx]
                    .shape()
                    .dims2()
                    .context("vision embedding must be 2D")?
                    .0,
                base_size,
                image_size,
                crop_mode,
            )?;
            tokens.extend(&placeholders);
            mask.extend(std::iter::repeat(1u8).take(placeholders.len()));
        }
    }

    let total_tokens = tokens.len();
    let image_tokens = mask.iter().filter(|&&flag| flag != 0).count();
    timer.finish(|event| {
        event.add_field("tokens", total_tokens);
        event.add_field("image_tokens", image_tokens);
        event.add_field("segments", segments.len());
        event.add_field("crop_mode", crop_mode);
    });

    Ok((tokens, mask))
}

/// Normalise decoder output by stripping sentinel tokens and Windows line-endings.
pub fn normalize_text(s: &str) -> String {
    s.replace("\r\n", "\n")
        .replace("<｜end▁of▁sentence｜>", "")
        .trim()
        .to_string()
}

fn build_image_placeholders(
    image_token_id: i64,
    input: &OwnedVisionInput,
    expected_tokens: usize,
    base_size: u32,
    image_size: u32,
    crop_mode: bool,
) -> Result<Vec<i64>> {
    const PATCH_SIZE: u32 = 16;
    const DOWNSAMPLE_RATIO: u32 = 4;

    let mut placeholders = Vec::new();

    let push_grid = |placeholders: &mut Vec<i64>, rows: usize, cols: usize, add_terminal: bool| {
        for _ in 0..rows {
            placeholders.extend(std::iter::repeat(image_token_id).take(cols));
            placeholders.push(image_token_id);
        }
        if add_terminal {
            placeholders.push(image_token_id);
        }
    };

    if crop_mode {
        let grid = (base_size / PATCH_SIZE) as usize;
        let num_queries_global = ((grid as f32) / (DOWNSAMPLE_RATIO as f32)).ceil() as usize;
        push_grid(
            &mut placeholders,
            num_queries_global,
            num_queries_global,
            true,
        );

        let (width_crops, height_crops) = input.crop_shape.unwrap_or((1, 1));
        if width_crops > 1 || height_crops > 1 {
            let local_grid = (image_size / PATCH_SIZE) as usize;
            let num_queries_local =
                ((local_grid as f32) / (DOWNSAMPLE_RATIO as f32)).ceil() as usize;
            let rows = num_queries_local * height_crops;
            let cols = num_queries_local * width_crops;
            push_grid(&mut placeholders, rows, cols, false);
        }
    } else {
        let grid = (image_size / PATCH_SIZE) as usize;
        let num_queries = ((grid as f32) / (DOWNSAMPLE_RATIO as f32)).ceil() as usize;
        push_grid(&mut placeholders, num_queries, num_queries, true);
    }

    anyhow::ensure!(
        placeholders.len() == expected_tokens,
        "placeholder count {} does not match expected {}",
        placeholders.len(),
        expected_tokens
    );
    Ok(placeholders)
}
