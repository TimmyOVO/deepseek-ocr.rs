use std::path::PathBuf;

use clap::{Args as ClapArgs, Parser, Subcommand};
use deepseek_ocr_pipeline::{DecodeParametersPatch, DeviceKind, OcrConfigPatch, Precision};

use crate::debug::DebugArgs;

#[derive(ClapArgs, Debug, Clone, Default)]
pub struct CommonModelArgs {
    /// Optional path to a configuration file (defaults to platform config dir).
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub config: Option<PathBuf>,

    /// Select model entry from configuration.
    #[arg(long, value_name = "ID", help_heading = "Application")]
    pub model: Option<String>,

    /// Override the model configuration JSON path.
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub model_config: Option<PathBuf>,

    /// Override tokenizer path.
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub tokenizer: Option<PathBuf>,

    /// Override weights path.
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub weights: Option<PathBuf>,
}

#[derive(ClapArgs, Debug, Clone, Default)]
pub struct CommonInferenceArgs {
    /// Device backend (cpu/metal/cuda).
    #[arg(long, help_heading = "Inference")]
    pub device: Option<DeviceKind>,

    /// Numeric precision override.
    #[arg(long, help_heading = "Inference")]
    pub dtype: Option<Precision>,

    /// Conversation template.
    #[arg(long, help_heading = "Inference")]
    pub template: Option<String>,

    /// Global view resolution.
    #[arg(long, help_heading = "Inference")]
    pub base_size: Option<u32>,

    /// Local crop resolution.
    #[arg(long, help_heading = "Inference")]
    pub image_size: Option<u32>,

    /// Enable dynamic crop mode.
    #[arg(long, help_heading = "Inference")]
    pub crop_mode: Option<bool>,

    /// Default max tokens budget.
    #[arg(long, help_heading = "Inference")]
    pub max_new_tokens: Option<usize>,

    /// Disable KV-cache usage during decoding.
    #[arg(long, help_heading = "Inference")]
    pub no_cache: bool,

    /// Enable sampling during decoding.
    #[arg(long, help_heading = "Inference", value_name = "BOOL")]
    pub do_sample: Option<bool>,

    /// Softmax temperature.
    #[arg(long, help_heading = "Inference")]
    pub temperature: Option<f64>,

    /// Nucleus sampling probability mass.
    #[arg(long, help_heading = "Inference")]
    pub top_p: Option<f64>,

    /// Top-k sampling cutoff.
    #[arg(long, help_heading = "Inference")]
    pub top_k: Option<usize>,

    /// Repetition penalty.
    #[arg(long, help_heading = "Inference")]
    pub repetition_penalty: Option<f32>,

    /// No-repeat n-gram size.
    #[arg(long, help_heading = "Inference")]
    pub no_repeat_ngram_size: Option<usize>,

    /// RNG seed.
    #[arg(long, help_heading = "Inference")]
    pub seed: Option<u64>,
}

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
    #[command(flatten)]
    pub model: CommonModelArgs,

    #[command(flatten)]
    pub inference: CommonInferenceArgs,

    /// Prompt text. Use `<image>` tokens to denote image slots.
    #[arg(long, conflicts_with = "prompt_file")]
    pub prompt: Option<String>,

    /// Prompt file path (UTF-8). Overrides `--prompt` when provided.
    #[arg(long, value_name = "PATH", conflicts_with = "prompt")]
    pub prompt_file: Option<PathBuf>,

    #[command(flatten)]
    pub debug: DebugArgs,

    /// Image files corresponding to `<image>` placeholders, in order.
    #[arg(long = "image", value_name = "PATH")]
    pub images: Vec<PathBuf>,

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

impl From<&InferArgs> for OcrConfigPatch {
    fn from(args: &InferArgs) -> Self {
        OcrConfigPatch {
            config_path: args.model.config.clone(),
            fs: None,
            model: deepseek_ocr_pipeline::OcrModelPatch {
                id: args
                    .model
                    .model
                    .as_deref()
                    .and_then(|value| deepseek_ocr_pipeline::OcrModelId::try_from(value).ok()),
                config: args.model.model_config.clone(),
                tokenizer: args.model.tokenizer.clone(),
                weights: args.model.weights.clone(),
                snapshot: None,
            },
            inference: deepseek_ocr_pipeline::OcrInferencePatch {
                device: args.inference.device,
                precision: args.inference.dtype,
                template: args.inference.template.clone(),
                vision: deepseek_ocr_pipeline::OcrVisionPatch {
                    base_size: args.inference.base_size,
                    image_size: args.inference.image_size,
                    crop_mode: args.inference.crop_mode,
                },
                decode: DecodeParametersPatch {
                    max_new_tokens: args.inference.max_new_tokens,
                    do_sample: args.inference.do_sample,
                    temperature: args.inference.temperature,
                    top_p: args.inference.top_p,
                    top_k: args.inference.top_k,
                    repetition_penalty: args.inference.repetition_penalty,
                    no_repeat_ngram_size: args.inference.no_repeat_ngram_size,
                    seed: args.inference.seed,
                    use_cache: args.inference.no_cache.then_some(false),
                },
            },
            server: deepseek_ocr_pipeline::OcrServerPatch::default(),
        }
    }
}
