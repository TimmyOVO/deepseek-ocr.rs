use std::path::PathBuf;

use clap::{Args as ClapArgs, Parser, Subcommand};
use deepseek_ocr_config::{AppConfig, ConfigOverride, ConfigOverrides};
use deepseek_ocr_core::runtime::{DeviceKind, Precision};

#[derive(Parser, Debug)]
#[command(author, version, about = "DeepSeek-OCR CLI", long_about = None)]
pub struct Cli {
    #[command(flatten)]
    pub infer: InferArgs,

    #[command(subcommand)]
    pub command: Option<CliCommand>,
}

#[derive(Subcommand, Debug)]
pub enum CliCommand {
    Weights(WeightsArgs),
}

#[derive(ClapArgs, Debug)]
pub struct WeightsArgs {
    #[command(subcommand)]
    pub command: WeightsCommand,
}

#[derive(Subcommand, Debug)]
pub enum WeightsCommand {
    #[command(name = "snapshot")]
    Snapshot(SnapshotArgs),
}

#[derive(ClapArgs, Debug)]
pub struct SnapshotArgs {
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub config: Option<PathBuf>,

    #[arg(long = "in", value_name = "PATH", help_heading = "Snapshot")]
    pub input: PathBuf,

    #[arg(long = "out", value_name = "PATH", help_heading = "Snapshot")]
    pub output: PathBuf,

    #[arg(
        long,
        value_name = "DTYPE",
        default_value = "Q8_0",
        value_parser = ["Q8_0", "Q4_K", "Q6_K"],
        help_heading = "Snapshot"
    )]
    pub dtype: String,

    #[arg(long, value_name = "TARGETS", default_value = "text", value_parser = ["text", "text+projector"], help_heading = "Snapshot")]
    pub targets: String,
}

#[derive(ClapArgs, Debug)]
pub struct InferArgs {
    /// Optional path to a configuration file (defaults to platform config dir).
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub config: Option<PathBuf>,

    /// Select which model entry to load from the configuration.
    #[arg(long, value_name = "ID", help_heading = "Application")]
    pub model: Option<String>,

    /// Override the model configuration JSON path.
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub model_config: Option<PathBuf>,

    /// Prompt text. Use `<image>` tokens to denote image slots.
    #[arg(long, conflicts_with = "prompt_file")]
    pub prompt: Option<String>,

    /// Prompt file path (UTF-8). Overrides `--prompt` when provided.
    #[arg(long, value_name = "PATH", conflicts_with = "prompt")]
    pub prompt_file: Option<PathBuf>,

    /// Conversation template name (plain/deepseek/deepseekv2/alignment).
    #[arg(long, help_heading = "Inference")]
    pub template: Option<String>,

    /// Image files corresponding to `<image>` placeholders, in order.
    #[arg(long = "image", value_name = "PATH")]
    pub images: Vec<PathBuf>,

    /// Override the default tokenizer path.
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub tokenizer: Option<PathBuf>,

    /// Override the weights path (defaults to DeepSeek-OCR/model-*.safetensors).
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub weights: Option<PathBuf>,

    /// Device backend to execute on (cpu/metal/cuda).
    #[arg(long, help_heading = "Inference")]
    pub device: Option<DeviceKind>,

    /// Numeric precision. Defaults to f32 on CPU and f16 on Metal/CUDA.
    #[arg(long, help_heading = "Inference")]
    pub dtype: Option<Precision>,

    /// Global view resolution (defaults to 1024).
    #[arg(long, help_heading = "Inference")]
    pub base_size: Option<u32>,

    /// Local crop resolution (defaults to 640).
    #[arg(long, help_heading = "Inference")]
    pub image_size: Option<u32>,

    /// Enable/disable dynamic crop mode (true/false).
    #[arg(long, help_heading = "Inference")]
    pub crop_mode: Option<bool>,

    /// Maximum number of tokens to generate.
    #[arg(long, help_heading = "Inference")]
    pub max_new_tokens: Option<usize>,

    /// Disable KV-cache usage during decoding.
    #[arg(long, help_heading = "Inference")]
    pub no_cache: bool,

    /// Enable sampling during decoding (true/false).
    #[arg(long, help_heading = "Inference", value_name = "BOOL")]
    pub do_sample: Option<bool>,

    /// Softmax temperature for sampling.
    #[arg(long, help_heading = "Inference")]
    pub temperature: Option<f64>,

    /// Nucleus sampling probability mass.
    #[arg(long, help_heading = "Inference")]
    pub top_p: Option<f64>,

    /// Top-k sampling cutoff.
    #[arg(long, help_heading = "Inference")]
    pub top_k: Option<usize>,

    /// Repetition penalty (>1 decreases repetition).
    #[arg(long, help_heading = "Inference")]
    pub repetition_penalty: Option<f32>,

    /// Enforce no-repeat n-gram constraint of the given size.
    #[arg(long, help_heading = "Inference")]
    pub no_repeat_ngram_size: Option<usize>,

    /// RNG seed for sampling.
    #[arg(long, help_heading = "Inference")]
    pub seed: Option<u64>,

    /// Enable benchmark instrumentation (requires `bench-metrics` feature).
    #[arg(long, help_heading = "Benchmark")]
    pub bench: bool,

    /// Write benchmark events to a JSON file.
    #[arg(long, value_name = "PATH", help_heading = "Benchmark")]
    pub bench_output: Option<PathBuf>,

    /// Quiet mode - output only the final result without logs or progress.
    #[arg(short, long, help_heading = "Application")]
    pub quiet: bool,
}

impl From<&InferArgs> for ConfigOverrides {
    fn from(args: &InferArgs) -> Self {
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
        overrides.inference.do_sample = args.do_sample;
        overrides.inference.temperature = args.temperature;
        overrides.inference.top_p = args.top_p;
        overrides.inference.top_k = args.top_k;
        overrides.inference.repetition_penalty = args.repetition_penalty;
        overrides.inference.no_repeat_ngram_size = args.no_repeat_ngram_size;
        overrides.inference.seed = args.seed;
        overrides
    }
}

impl ConfigOverride for &InferArgs {
    fn apply(self, config: &mut AppConfig) {
        config.apply_overrides(&ConfigOverrides::from(self));
    }
}
