use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Tensor};
use clap::Parser;
use deepseek_ocr_assets as assets;
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
    /// Prompt text. Use `<image>` tokens to denote image slots.
    #[arg(long, conflicts_with = "prompt_file")]
    prompt: Option<String>,

    /// Prompt file path (UTF-8). Overrides `--prompt` when provided.
    #[arg(long, value_name = "PATH", conflicts_with = "prompt")]
    prompt_file: Option<PathBuf>,

    /// Conversation template name (plain/deepseek/deepseekv2/alignment).
    #[arg(long, default_value = "plain")]
    template: String,

    /// Image files corresponding to `<image>` placeholders, in order.
    #[arg(long = "image", value_name = "PATH")]
    images: Vec<PathBuf>,

    /// Override the default tokenizer path.
    #[arg(
        long,
        default_value = assets::DEFAULT_TOKENIZER_PATH,
        value_name = "PATH"
    )]
    tokenizer: PathBuf,

    /// Override the weights path (defaults to DeepSeek-OCR/model-*.safetensors).
    #[arg(long, value_name = "PATH")]
    weights: Option<PathBuf>,

    /// Device backend to execute on (cpu/metal/cuda).
    #[arg(long, default_value = "cpu")]
    device: DeviceKind,

    /// Numeric precision. Defaults to f32 on CPU and f16 on Metal/CUDA.
    #[arg(long)]
    dtype: Option<Precision>,

    /// Global view resolution (defaults to 1024).
    #[arg(long, default_value_t = 1024)]
    base_size: u32,

    /// Local crop resolution (defaults to 640).
    #[arg(long, default_value_t = 640)]
    image_size: u32,

    /// Enable/disable dynamic crop mode (true/false).
    #[arg(long, default_value_t = true)]
    crop_mode: bool,

    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,

    /// Disable KV-cache usage during decoding.
    #[arg(long, default_value_t = false)]
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

    let config_path = assets::ensure_config()?;
    let weights_path = assets::resolve_weights(args.weights.as_deref())?;

    let (device, maybe_precision) =
        prepare_device_and_dtype(args.device.clone(), args.dtype.clone())?;
    let dtype = maybe_precision.unwrap_or_else(|| default_dtype_for_device(&device));

    println!(
        "Loading model (device={:?}, dtype={:?}) from {}",
        device,
        dtype,
        weights_path.display()
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
        "Model ready in {:.2?} (flash-attn: {})",
        load_start.elapsed(),
        model.flash_attention_enabled()
    );

    let tokenizer_path = assets::ensure_tokenizer(&args.tokenizer)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
        anyhow!(
            "failed to load tokenizer from {}: {err}",
            tokenizer_path.display()
        )
    })?;

    let prompt_with_template = render_prompt(&args.template, "", &prompt_raw)?;
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
        args.base_size,
        args.image_size,
        args.crop_mode,
    )?;
    let embeddings = compute_image_embeddings(&model, &owned_inputs)?;

    let (input_ids_vec, mask_vec) = build_prompt_tokens(
        &tokenizer,
        &prompt_with_template,
        &embeddings,
        &owned_inputs,
        args.base_size,
        args.image_size,
        args.crop_mode,
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

    let mut options = GenerateOptions::new(args.max_new_tokens);
    options.images_seq_mask = Some(&mask_tensor);
    if !embeddings.is_empty() {
        options.image_embeddings = Some(embeddings.as_slice());
    }
    options.eos_token_id = model.language_model().config().eos_token_id;
    if args.no_cache {
        options.use_cache = false;
    }

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
        args.max_new_tokens
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
