use std::{
    cell::{Cell, RefCell},
    convert::TryFrom,
    io::{self, Write},
    rc::Rc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use deepseek_ocr_config::{AppConfig, LocalFileSystem};
use deepseek_ocr_core::{
    ModelKind, ModelLoadArgs,
    inference::{DecodeOutcome, DecodeParameters, VisionSettings, render_prompt},
    runtime::{default_dtype_for_device, prepare_device_and_dtype},
    streaming::DeltaTracker,
};
use deepseek_ocr_infer_deepseek::{
    load_model as load_deepseek_model,
    quant_snapshot::{SNAPSHOT_SPEC_PATH, qtensor_bytes_supported},
};
use deepseek_ocr_infer_dots::load_model as load_dots_model;
use deepseek_ocr_infer_paddleocr::load_model as load_paddle_model;
use image::DynamicImage;
use tokenizers::Tokenizer;
use tracing::info;

use crate::{
    args::{InferArgs, SnapshotArgs, WeightsArgs, WeightsCommand},
    bench,
    prompt::load_prompt,
    resources::{
        ensure_config_file, ensure_tokenizer_file, prepare_snapshot_path, prepare_weights_path,
    },
};

#[derive(Default)]
struct StreamProgress {
    last_count: usize,
    delta: DeltaTracker,
}

pub fn run_inference(args: InferArgs) -> Result<()> {
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

    let config_path = ensure_config_file(&fs, &resources.id, &resources.config)?;
    let tokenizer_path = ensure_tokenizer_file(&fs, &resources.id, &resources.tokenizer)?;
    let weights_path = prepare_weights_path(&fs, &resources.id, &resources.weights)?;
    let snapshot_path = prepare_snapshot_path(&fs, &resources.id, resources.snapshot.as_ref())?;

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
    let load_args = ModelLoadArgs {
        kind: resources.kind,
        config_path: Some(&config_path),
        weights_path: Some(&weights_path),
        snapshot_path: snapshot_path.as_deref(),
        device: device.clone(),
        dtype,
    };
    let model = match resources.kind {
        ModelKind::Deepseek => load_deepseek_model(load_args)?,
        ModelKind::PaddleOcrVl => load_paddle_model(load_args)?,
        ModelKind::DotsOcr => load_dots_model(load_args)?,
    };
    let load_elapsed = load_start.elapsed();
    info!(
        "Model ready in {:.2?} (kind={:?}, flash-attn: {}, weights={})",
        load_elapsed,
        model.kind(),
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

    let vision_settings = VisionSettings {
        base_size: app_config.inference.base_size,
        image_size: app_config.inference.image_size,
        crop_mode: app_config.inference.crop_mode,
    };
    let decode_params = DecodeParameters {
        max_new_tokens: app_config.inference.max_new_tokens,
        do_sample: app_config.inference.do_sample,
        temperature: app_config.inference.temperature,
        top_p: if app_config.inference.top_p < 1.0 {
            Some(app_config.inference.top_p)
        } else {
            None
        },
        top_k: app_config.inference.top_k,
        repetition_penalty: app_config.inference.repetition_penalty,
        no_repeat_ngram_size: app_config.inference.no_repeat_ngram_size,
        seed: app_config.inference.seed,
        use_cache: app_config.inference.use_cache,
    };

    let tokenizer_for_stream = tokenizer.clone();
    let progress_state = Rc::new(RefCell::new(StreamProgress::default()));
    let stream_state = Rc::clone(&progress_state);
    let start_time_cell = Rc::new(Cell::new(None::<Instant>));
    let prefill_duration_cell = Rc::new(Cell::new(None::<Duration>));
    let start_time_for_cb = Rc::clone(&start_time_cell);
    let prefill_duration_for_cb = Rc::clone(&prefill_duration_cell);
    let stdout = Rc::new(RefCell::new(io::stdout()));
    let stdout_handle = Rc::clone(&stdout);
    let progress_callback = move |count: usize, ids: &[i64]| {
        let mut delta_to_emit = None;

        if count > 0 && prefill_duration_for_cb.get().is_none() {
            if let Some(start) = start_time_for_cb.get() {
                prefill_duration_for_cb.set(Some(start.elapsed()));
            }
        }

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

    let mut callback_holder: Option<Box<dyn Fn(usize, &[i64])>> = None;
    if !quiet {
        callback_holder = Some(Box::new(progress_callback));
    }

    info!(
        "Starting generation with requested budget {} tokens",
        app_config.inference.max_new_tokens
    );
    info!("--- Generation start ---");
    let gen_start = Instant::now();
    start_time_cell.set(Some(gen_start));
    let outcome = model
        .decode(
            &tokenizer,
            &prompt_with_template,
            &images,
            vision_settings,
            &decode_params,
            callback_holder.as_deref(),
        )
        .context("generation failed")?;
    let elapsed = gen_start.elapsed();
    info!("--- Generation done in {:.2?} ---", elapsed);

    let DecodeOutcome {
        text: normalized,
        prompt_tokens,
        response_tokens,
        generated_tokens,
    } = outcome;

    info!(
        "Prompt prepared: {} tokens ({} image slots)",
        prompt_tokens,
        args.images.len()
    );

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
    info!("Final output:\n{normalized}");

    {
        let total_elapsed = elapsed;
        let prefill_elapsed = prefill_duration_cell
            .get()
            .filter(|duration| *duration <= total_elapsed)
            .unwrap_or(total_elapsed);
        let decode_elapsed = total_elapsed
            .checked_sub(prefill_elapsed)
            .unwrap_or_default();
        let generated_count = response_tokens;
        let prefill_secs = prefill_elapsed.as_secs_f64();
        let decode_secs = decode_elapsed.as_secs_f64();
        let prefill_rate = if prefill_secs > 0.0 {
            prompt_tokens as f64 / prefill_secs
        } else {
            0.0
        };
        let decode_rate = if decode_secs > 0.0 {
            generated_count as f64 / decode_secs
        } else {
            0.0
        };
        info!(
            "Throughput: prefill={prompt_tokens} tok in {prefill_secs:.2}s ({prefill_rate:.2} tok/s); generation={generated_count} tok in {decode_secs:.2}s ({decode_rate:.2} tok/s)"
        );
    }

    if let Some(session) = bench_session {
        let report = session.finalize()?;
        bench::print_summary(&report);
    }

    Ok(())
}

pub fn run_weights(args: WeightsArgs) -> Result<()> {
    match args.command {
        WeightsCommand::Snapshot(cmd) => run_snapshot(cmd),
    }
}

fn run_snapshot(cmd: SnapshotArgs) -> Result<()> {
    let mut instructions = vec![
        "cargo run -p deepseek-ocr-dsq-cli --release -- export".to_string(),
        format!("--weights {}", cmd.input.display()),
        format!("--output {}", cmd.output.display()),
        format!("--dtype {}", cmd.dtype),
        format!("--targets {}", cmd.targets),
    ];
    if let Some(config) = cmd.config.as_ref() {
        instructions.push(format!("--config {}", config.display()));
    }
    let command = instructions.join(" ");
    let prefix = if qtensor_bytes_supported() {
        "Snapshot export inside the runtime is waiting on Candle QTensor byte APIs."
    } else {
        "Runtime snapshot export depends on upcoming Candle QTensor serialization support."
    };
    bail!(
        "{prefix}\nUse `{command}` to build the .dsq container via the Rust exporter. Design reference: {spec}",
        prefix = prefix,
        command = command,
        spec = SNAPSHOT_SPEC_PATH
    );
}
