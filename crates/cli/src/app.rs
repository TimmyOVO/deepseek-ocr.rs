use std::{
    cell::{Cell, RefCell},
    convert::TryFrom,
    io::{self, Write},
    path::PathBuf,
    rc::Rc,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use deepseek_ocr_pipeline::{
    DeviceKind, ModelKind, OcrConfigResolver, OcrConfigSource, OcrPatchLayer, OcrPipelineEvent,
    OcrPipelineObserver, OcrPrompt, OcrRequest, OcrRuntimeBuilder, Precision,
};
use image::DynamicImage;
use tokenizers::Tokenizer;
use tracing::info;

use crate::{
    args::{InferArgs, SnapshotArgs, WeightsArgs, WeightsCommand},
    bench, debug,
    prompt::load_prompt,
};

type TokenCallback = Box<dyn Fn(usize, &[i64])>;
const SNAPSHOT_SPEC_PATH: &str = "docs/quant_snapshot.md";

const fn qtensor_bytes_supported() -> bool {
    false
}

#[derive(Default)]
struct StreamProgress {
    last_count: usize,
    emitted_text: String,
}

#[derive(Default, Clone)]
struct LoadMetadataObserver {
    state: Arc<Mutex<LoadMetadata>>,
}

#[derive(Default, Clone)]
struct LoadMetadata {
    model_id: Option<String>,
    model_kind: Option<ModelKind>,
    flash_attention: Option<bool>,
    config_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    weights_path: Option<PathBuf>,
}

impl LoadMetadataObserver {
    fn snapshot(&self) -> LoadMetadata {
        self.state.lock().map(|state| state.clone()).unwrap_or_default()
    }
}

impl OcrPipelineObserver for LoadMetadataObserver {
    fn on_event(&self, event: &OcrPipelineEvent) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match event {
            OcrPipelineEvent::ResourcesPrepared {
                model_id,
                config,
                tokenizer,
                weights,
                ..
            } => {
                state.model_id = Some(model_id.to_string());
                state.config_path = Some(PathBuf::from(config));
                state.tokenizer_path = Some(PathBuf::from(tokenizer));
                state.weights_path = Some(PathBuf::from(weights));
            }
            OcrPipelineEvent::ModelLoadFinished {
                kind,
                flash_attention,
                ..
            } => {
                state.model_kind = Some(*kind);
                state.flash_attention = Some(*flash_attention);
            }
            _ => {}
        }
    }
}

const fn effective_dtype_label(device: DeviceKind, precision: Option<Precision>) -> &'static str {
    match precision {
        Some(Precision::F16) => "F16",
        Some(Precision::F32) => "F32",
        Some(Precision::Bf16) => "BF16",
        None => match device {
            DeviceKind::Cpu => "F32",
            DeviceKind::Metal | DeviceKind::Cuda => "F16",
        },
    }
}

fn stream_delta(previous: &str, current: &str) -> String {
    if let Some(suffix) = current.strip_prefix(previous) {
        return suffix.to_owned();
    }
    let prefix_len = previous
        .chars()
        .zip(current.chars())
        .take_while(|(lhs, rhs)| lhs == rhs)
        .map(|(ch, _)| ch.len_utf8())
        .sum::<usize>();
    current[prefix_len..].to_owned()
}

pub fn run_inference(args: InferArgs) -> Result<()> {
    let quiet = args.quiet;
    let bench_enabled = args.bench || args.bench_output.is_some();
    let bench_session = bench::maybe_start(bench_enabled, args.bench_output.clone())?;

    // When `cli-debug` is enabled, allow overriding the prompt with a baseline
    // `prompt.json` (rendered_prompt). This must happen before we validate
    // that a prompt is present.
    let prompt_raw = if let Some(override_prompt) = debug::load_prompt_override(&args.debug)? {
        override_prompt
    } else {
        load_prompt(&args)?
    };

    if let Some(model_id) = args.model.model.as_deref() {
        deepseek_ocr_pipeline::OcrModelId::try_from(model_id)
            .with_context(|| format!("invalid model id `{model_id}`"))?;
    }

    let cli_patch = deepseek_ocr_pipeline::OcrConfigPatch::from(&args);
    let mut resolver = OcrConfigResolver::new();
    resolver.push_layer(OcrPatchLayer::new(OcrConfigSource::CliArgs, cli_patch.clone()));
    let app_config = resolver.resolve()?;

    info!(
        "Using configuration {} (active model `{}`)",
        args.model
            .config
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<platform-default>".to_string()),
        app_config.models.active
    );

    let observer = LoadMetadataObserver::default();
    let runtime = OcrRuntimeBuilder::new()
        .with_observer(Arc::new(observer.clone()))
        .with_cli_args_layer(cli_patch)
        .build()?;
    let manager = runtime.manager();
    let device = manager.device_kind();
    let dtype = effective_dtype_label(device, manager.precision());
    let model_id = manager
        .active_model_id()
        .map(ToString::to_string)
        .unwrap_or_else(|| app_config.models.active.clone());

    info!(
        "Loading model `{}` (device={:?}, dtype={})",
        model_id,
        device,
        dtype,
    );

    let load_start = Instant::now();
    let pipeline_handle = manager.load_active()?;
    let load_elapsed = load_start.elapsed();
    let metadata = observer.snapshot();
    let model_kind = metadata.model_kind.unwrap_or(ModelKind::Deepseek);
    let flash_attention = metadata.flash_attention.unwrap_or(false);
    let weights_path = metadata.weights_path.unwrap_or_default();
    info!(
        "Model ready in {:.2?} (kind={:?}, flash-attn: {}, weights={})",
        load_elapsed,
        model_kind,
        flash_attention,
        weights_path.display()
    );

    let prompt_user = prompt_raw.clone();

    let images: Vec<DynamicImage> = args
        .images
        .iter()
        .map(|path| {
            image::open(path).with_context(|| format!("failed to open image at {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;

    let vision_settings = app_config.inference.to_vision_settings();
    let decode_params = app_config.inference.decode.clone();
    let request = OcrRequest {
        prompt: OcrPrompt::Raw(prompt_raw.clone()),
        template: app_config.inference.template.clone(),
        system_prompt: String::new(),
        images,
        vision: vision_settings,
        decode: decode_params,
    };

    let tokenizer: Tokenizer = pipeline_handle.pipeline().tokenizer()?.clone();

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

        if count > 0
            && prefill_duration_for_cb.get().is_none()
            && let Some(start) = start_time_for_cb.get()
        {
            prefill_duration_for_cb.set(Some(start.elapsed()));
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
                let delta = stream_delta(&state.emitted_text, &full_text);
                if !delta.is_empty() {
                    delta_to_emit = Some(delta);
                }
                state.emitted_text = full_text;
            }

            state.last_count = count;
        }

        if let Some(delta) = delta_to_emit {
            let mut handle = stdout_handle.borrow_mut();
            let _ = write!(handle, "{}", delta);
            let _ = handle.flush();
        }
    };

    let mut callback_holder: Option<TokenCallback> = None;
    if !quiet {
        callback_holder = Some(Box::new(progress_callback));
    }

    info!(
        "Starting generation with requested budget {} tokens",
        request.decode.max_new_tokens
    );
    info!("--- Generation start ---");
    let gen_start = Instant::now();
    start_time_cell.set(Some(gen_start));
    let outcome = pipeline_handle
        .generate(&request, None, callback_holder.as_deref())
        .context("generation failed")?;
    let elapsed = gen_start.elapsed();
    info!("--- Generation done in {:.2?} ---", elapsed);

    let normalized = outcome.text;
    let prompt_tokens = outcome.prompt_tokens;
    let response_tokens = outcome.response_tokens;
    let generated_tokens = outcome.generated_tokens;
    let rendered_prompt = outcome.rendered_prompt;

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

    if debug::wants_output_json(&args.debug) {
        let device_label = format!("{device:?}");
        let dtype_label = dtype.to_string();
        let image_paths: Vec<String> = args
            .images
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        let model_id = metadata.model_id.unwrap_or(model_id);
        let tokenizer_path = metadata.tokenizer_path.unwrap_or_default();
        debug::write_output_json(
            &args.debug,
            debug::DebugOutput {
                model_id: &model_id,
                weights_path: &weights_path,
                tokenizer_path: &tokenizer_path,
                device: &device_label,
                dtype: &dtype_label,
                template: &app_config.inference.template,
                base_size: app_config.inference.base_size,
                image_size: app_config.inference.image_size,
                crop_mode: app_config.inference.crop_mode,
                max_new_tokens: app_config.inference.decode.max_new_tokens,
                repetition_penalty: app_config.inference.decode.repetition_penalty,
                no_repeat_ngram_size: app_config.inference.decode.no_repeat_ngram_size,
                use_cache: app_config.inference.decode.use_cache,
                prompt_user: &prompt_user,
                rendered_prompt: &rendered_prompt,
                image_paths: &image_paths,
                prompt_tokens,
                generated_len: response_tokens,
                tokens: &generated_tokens,
                decoded: &decoded,
                normalized: &normalized,
            },
        )?;
    }

    // When quiet, we must not emit any decoded text to stdout.
    // The gate script relies on stdout being clean for progress visualization.
    if !quiet {
        let final_delta = {
            let mut state = progress_state.borrow_mut();
            state.last_count = generated_tokens.len();
            let delta = stream_delta(&state.emitted_text, &decoded);
            state.emitted_text = decoded.clone();
            delta
        };
        if !final_delta.is_empty() {
            let mut handle = stdout.borrow_mut();
            let _ = write!(handle, "{}", final_delta);
            let _ = handle.flush();
        }
        info!("Final output:\n{normalized}");
    }
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
