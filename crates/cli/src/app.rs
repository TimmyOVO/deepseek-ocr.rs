use std::{
    cell::RefCell,
    convert::TryFrom,
    io::{self, Write},
    rc::Rc,
    time::Instant,
};

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use deepseek_ocr_config::{AppConfig, LocalFileSystem};
use deepseek_ocr_core::{
    inference::{
        build_prompt_tokens, compute_image_embeddings, normalize_text, prepare_vision_inputs,
        render_prompt,
    },
    model::{DeepseekOcrModel, GenerateOptions},
    runtime::{default_dtype_for_device, prepare_device_and_dtype},
    streaming::DeltaTracker,
};
use image::DynamicImage;
use tokenizers::Tokenizer;
use tracing::info;

use crate::{
    args::Args,
    bench,
    prompt::load_prompt,
    resources::{ensure_config_file, ensure_tokenizer_file, prepare_weights_path},
};

#[derive(Default)]
struct StreamProgress {
    last_count: usize,
    delta: DeltaTracker,
}

pub fn run(args: Args) -> Result<()> {
    let quiet = args.quiet;
    let bench_enabled = args.bench || args.bench_output.is_some();
    let bench_session = bench::maybe_start(bench_enabled, args.bench_output.clone())?;

    let prompt_raw = load_prompt(&args)?;

    let fs = LocalFileSystem::new("deepseek-ocr");
    let (mut app_config, descriptor) = AppConfig::load_or_init(&fs, args.config.as_deref())?;
    app_config += &args;
    app_config.normalise(&fs)?;
    let resources = app_config.active_model_resources(&fs)?;

    info!(
        "Using configuration {} (active model `{}`)",
        descriptor.location.display_with(&fs)?,
        app_config.models.active
    );

    let config_path = ensure_config_file(&fs, &resources.config)?;
    let tokenizer_path = ensure_tokenizer_file(&fs, &resources.tokenizer)?;
    let weights_path = prepare_weights_path(&fs, &resources.weights)?;

    let (device, maybe_precision) =
        prepare_device_and_dtype(app_config.inference.device, app_config.inference.precision)?;
    let dtype = maybe_precision.unwrap_or_else(|| default_dtype_for_device(&device));

    info!(
        "Loading model `{}` (device={:?}, dtype={:?}) using config {}",
        app_config.models.active,
        device,
        dtype,
        config_path.display()
    );

    let load_start = Instant::now();
    let model = DeepseekOcrModel::load(
        Some(&config_path),
        Some(&weights_path),
        device.clone(),
        dtype,
    )
    .context("failed to load DeepSeek-OCR model")?;
    info!(
        "Model ready in {:.2?} (flash-attn: {}, weights={})",
        load_start.elapsed(),
        model.flash_attention_enabled(),
        weights_path.display()
    );

    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
        anyhow::anyhow!(
            "failed to load tokenizer from {}: {err}",
            tokenizer_path.display()
        )
    })?;

    let prompt_with_template = render_prompt(&app_config.inference.template, "", &prompt_raw)?;
    let image_slots = prompt_with_template.matches("<image>").count();
    anyhow::ensure!(
        image_slots == args.images.len(),
        "prompt includes {image_slots} <image> tokens but {} image paths were provided",
        args.images.len()
    );

    let images: Vec<DynamicImage> = args
        .images
        .iter()
        .map(|path| {
            image::open(path).with_context(|| format!("failed to open image at {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;

    let owned_inputs = prepare_vision_inputs(
        &model,
        &images,
        app_config.inference.base_size,
        app_config.inference.image_size,
        app_config.inference.crop_mode,
    )?;
    let embeddings = compute_image_embeddings(&model, &owned_inputs)?;

    let (input_ids_vec, mask_vec) = build_prompt_tokens(
        &tokenizer,
        &prompt_with_template,
        &embeddings,
        &owned_inputs,
        app_config.inference.base_size,
        app_config.inference.image_size,
        app_config.inference.crop_mode,
    )?;

    info!(
        "Prompt prepared: {} tokens ({} image tokens)",
        input_ids_vec.len(),
        mask_vec.iter().filter(|&&b| b != 0).count()
    );

    let input_ids = Tensor::from_vec(
        input_ids_vec.clone(),
        (1, input_ids_vec.len()),
        model.device(),
    )?
    .to_dtype(DType::I64)?;
    let mask_tensor = Tensor::from_vec(mask_vec.clone(), (1, mask_vec.len()), model.device())?
        .to_dtype(DType::U8)?;

    let mut options = GenerateOptions::new(app_config.inference.max_new_tokens);
    options.images_seq_mask = Some(&mask_tensor);
    if !embeddings.is_empty() {
        options.image_embeddings = Some(embeddings.as_slice());
    }
    options.eos_token_id = model.language_model().config().eos_token_id;
    options.use_cache = app_config.inference.use_cache;

    let tokenizer_for_stream = tokenizer.clone();
    let progress_state = Rc::new(RefCell::new(StreamProgress::default()));
    let stream_state = Rc::clone(&progress_state);
    let stdout = Rc::new(RefCell::new(io::stdout()));
    let stdout_handle = Rc::clone(&stdout);
    let progress_callback = move |count: usize, ids: &[i64]| {
        let mut delta_to_emit = None;

        {
            let mut state = stream_state.borrow_mut();
            if count <= state.last_count {
                state.last_count = count;
                return;
            }

            let token_slice: Vec<u32> = ids[..count]
                .iter()
                .filter_map(|&id| u32::try_from(id).ok())
                .collect();

            if token_slice.is_empty() {
                state.last_count = count;
                return;
            }

            if let Ok(full_text) = tokenizer_for_stream.decode(&token_slice, true) {
                let delta = state.delta.advance(&full_text, false);
                if !delta.is_empty() {
                    delta_to_emit = Some(delta);
                }
            }

            state.last_count = count;
        }

        if let Some(delta) = delta_to_emit {
            let mut handle = stdout_handle.borrow_mut();
            let _ = write!(handle, "{}", delta);
            let _ = handle.flush();
        }
    };

    // Only enable streaming progress if not in quiet mode
    if !quiet {
        options.progress_callback = Some(&progress_callback);
    }

    info!(
        "Starting generation with requested budget {} tokens",
        app_config.inference.max_new_tokens
    );
    info!("--- Generation start ---");
    let gen_start = Instant::now();
    let generated = model.generate(&input_ids, options)?;
    let elapsed = gen_start.elapsed();
    info!("--- Generation done in {:.2?} ---", elapsed);

    let generated_tokens = generated
        .to_vec2::<i64>()?
        .into_iter()
        .next()
        .unwrap_or_default();
    let decoded = tokenizer
        .decode(
            &generated_tokens
                .iter()
                .filter_map(|&id| u32::try_from(id).ok())
                .collect::<Vec<_>>(),
            true,
        )
        .unwrap_or_default();

    let final_delta = {
        let mut state = progress_state.borrow_mut();
        state.last_count = generated_tokens.len();
        state.delta.advance(&decoded, true)
    };
    if !final_delta.is_empty() {
        let mut handle = stdout.borrow_mut();
        let _ = write!(handle, "{}", final_delta);
        let _ = handle.flush();
    }
    let normalized = normalize_text(&decoded);
    info!("Final output:\n{normalized}");

    if let Some(session) = bench_session {
        let report = session.finalize()?;
        bench::print_summary(&report);
    }

    Ok(())
}
