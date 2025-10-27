use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Tensor};
use clap::Parser;
use deepseek_ocr_assets as assets;
use deepseek_ocr_config::{
    AppConfig, ConfigOverride, ConfigOverrides, LocalFileSystem, ResourceLocation,
    VirtualFileSystem,
};
use deepseek_ocr_core::{
    inference::{
        build_prompt_tokens, compute_image_embeddings, normalize_text, prepare_vision_inputs,
        render_prompt,
    },
    model::{DeepseekOcrModel, GenerateOptions},
    runtime::{DeviceKind, Precision, default_dtype_for_device, prepare_device_and_dtype},
};
use image::DynamicImage;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about = "DeepSeek-OCR CLI", long_about = None)]
struct Args {
    /// Optional path to a configuration file (defaults to platform config dir).
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    config: Option<PathBuf>,

    /// Select which model entry to load from the configuration.
    #[arg(long, value_name = "ID", help_heading = "Application")]
    model: Option<String>,

    /// Override the model configuration JSON path.
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    model_config: Option<PathBuf>,

    /// Prompt text. Use `<image>` tokens to denote image slots.
    #[arg(long, conflicts_with = "prompt_file")]
    prompt: Option<String>,

    /// Prompt file path (UTF-8). Overrides `--prompt` when provided.
    #[arg(long, value_name = "PATH", conflicts_with = "prompt")]
    prompt_file: Option<PathBuf>,

    /// Conversation template name (plain/deepseek/deepseekv2/alignment).
    #[arg(long, help_heading = "Inference")]
    template: Option<String>,

    /// Image files corresponding to `<image>` placeholders, in order.
    #[arg(long = "image", value_name = "PATH")]
    images: Vec<PathBuf>,

    /// Override the default tokenizer path.
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    tokenizer: Option<PathBuf>,

    /// Override the weights path (defaults to DeepSeek-OCR/model-*.safetensors).
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    weights: Option<PathBuf>,

    /// Device backend to execute on (cpu/metal/cuda).
    #[arg(long, help_heading = "Inference")]
    device: Option<DeviceKind>,

    /// Numeric precision. Defaults to f32 on CPU and f16 on Metal/CUDA.
    #[arg(long, help_heading = "Inference")]
    dtype: Option<Precision>,

    /// Global view resolution (defaults to 1024).
    #[arg(long, help_heading = "Inference")]
    base_size: Option<u32>,

    /// Local crop resolution (defaults to 640).
    #[arg(long, help_heading = "Inference")]
    image_size: Option<u32>,

    /// Enable/disable dynamic crop mode (true/false).
    #[arg(long, help_heading = "Inference")]
    crop_mode: Option<bool>,

    /// Maximum number of tokens to generate.
    #[arg(long, help_heading = "Inference")]
    max_new_tokens: Option<usize>,

    /// Disable KV-cache usage during decoding.
    #[arg(long, help_heading = "Inference")]
    no_cache: bool,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let prompt_raw = load_prompt(&args)?;

    let fs = LocalFileSystem::new("deepseek-ocr");
    let (mut app_config, descriptor) = AppConfig::load_or_init(&fs, args.config.as_deref())?;
    app_config += &args;
    app_config.normalise(&fs)?;
    let resources = app_config.active_model_resources(&fs)?;

    println!(
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

    println!(
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
    println!(
        "Model ready in {:.2?} (flash-attn: {}, weights={})",
        load_start.elapsed(),
        model.flash_attention_enabled(),
        weights_path.display()
    );

    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
        anyhow!(
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

    println!(
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
    let progress_state = std::rc::Rc::new(std::cell::RefCell::new(0usize));
    let stream_state = std::rc::Rc::clone(&progress_state);
    let stdout = std::rc::Rc::new(std::cell::RefCell::new(io::stdout()));
    let stdout_handle = std::rc::Rc::clone(&stdout);
    let progress_callback = move |count: usize, ids: &[i64]| {
        let mut last = stream_state.borrow_mut();
        if count <= *last {
            return;
        }
        let new_tokens: Vec<u32> = ids[*last..count]
            .iter()
            .filter_map(|&id| u32::try_from(id).ok())
            .collect();
        if !new_tokens.is_empty() {
            if let Ok(decoded) = tokenizer_for_stream.decode(&new_tokens, true) {
                if !decoded.is_empty() {
                    let mut handle = stdout_handle.borrow_mut();
                    let _ = write!(handle, "{}", decoded);
                    let _ = handle.flush();
                }
            }
        }
        *last = count;
    };
    options.progress_callback = Some(&progress_callback);

    println!(
        "Starting generation with requested budget {} tokens.",
        app_config.inference.max_new_tokens
    );
    println!("\n--- Generation start ---\n");
    let gen_start = Instant::now();
    let generated = model.generate(&input_ids, options)?;
    let elapsed = gen_start.elapsed();
    println!("\n\n--- Generation done in {:.2?} ---", elapsed);

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
    let normalized = normalize_text(&decoded);
    println!("\nFinal output:\n{}", normalized);

    Ok(())
}

fn ensure_config_file(fs: &LocalFileSystem, location: &ResourceLocation) -> Result<PathBuf> {
    ensure_resource(fs, location, |path| assets::ensure_config_at(path))
}

fn ensure_tokenizer_file(fs: &LocalFileSystem, location: &ResourceLocation) -> Result<PathBuf> {
    ensure_resource(fs, location, |path| assets::ensure_tokenizer_at(path))
}

fn prepare_weights_path(fs: &LocalFileSystem, location: &ResourceLocation) -> Result<PathBuf> {
    ensure_resource(fs, location, |path| {
        assets::resolve_weights_with_default(None, path)
    })
}

impl From<&Args> for ConfigOverrides {
    fn from(args: &Args) -> Self {
        let mut overrides = ConfigOverrides::default();
        overrides.config_path = args.config.clone();
        overrides.model_id = args.model.clone();
        overrides.model_config = args.model_config.clone();
        overrides.tokenizer = args.tokenizer.clone();
        overrides.weights = args.weights.clone();
        overrides.inference.device = args.device;
        overrides.inference.precision = args.dtype;
        overrides.inference.template = args.template.clone();
        overrides.inference.base_size = args.base_size;
        overrides.inference.image_size = args.image_size;
        overrides.inference.crop_mode = args.crop_mode;
        overrides.inference.max_new_tokens = args.max_new_tokens;
        if args.no_cache {
            overrides.inference.use_cache = Some(false);
        }
        overrides
    }
}

impl ConfigOverride for &Args {
    fn apply(self, config: &mut AppConfig) {
        config.apply_overrides(&ConfigOverrides::from(self));
    }
}

fn ensure_resource<F>(
    fs: &LocalFileSystem,
    location: &ResourceLocation,
    ensure_fn: F,
) -> Result<PathBuf>
where
    F: Fn(&Path) -> Result<PathBuf>,
{
    match location {
        ResourceLocation::Physical(path) => ensure_fn(path),
        ResourceLocation::Virtual(vpath) => fs.with_physical_path(vpath, |physical| {
            let resolved = ensure_fn(physical)?;
            Ok(resolved)
        }),
    }
}

fn load_prompt(args: &Args) -> Result<String> {
    if let Some(path) = &args.prompt_file {
        return fs::read_to_string(path)
            .with_context(|| format!("failed to read prompt file {}", path.display()))
            .map(|s| s.trim_end().to_owned());
    }
    if let Some(prompt) = &args.prompt {
        return Ok(prompt.clone());
    }
    Err(anyhow!(
        "prompt is required (use --prompt or --prompt-file)"
    ))
}
