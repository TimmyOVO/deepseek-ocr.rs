use std::path::PathBuf;

use clap::Parser;
use deepseek_ocr_config::{AppConfig, ConfigOverride, ConfigOverrides};
use deepseek_ocr_core::runtime::{DeviceKind, Precision};

#[derive(Parser, Debug)]
#[command(author, version, about = "DeepSeek-OCR API Server", long_about = None)]
pub struct Args {
    /// Optional path to a configuration file (defaults to platform config dir).
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub config: Option<PathBuf>,

    /// Select the model entry to serve (configuration file).
    #[arg(long, value_name = "ID", help_heading = "Application")]
    pub model: Option<String>,

    /// Override the model configuration JSON path.
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub model_config: Option<PathBuf>,

    /// Tokenizer path.
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub tokenizer: Option<PathBuf>,

    /// Optional weights override (defaults to DeepSeek-OCR/model-*.safetensors).
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub weights: Option<PathBuf>,

    /// Device backend (cpu/metal/cuda).
    #[arg(long, help_heading = "Inference")]
    pub device: Option<DeviceKind>,

    /// Numeric precision override (cpu=f32 default, metal/cuda=f16).
    #[arg(long, help_heading = "Inference")]
    pub dtype: Option<Precision>,

    /// Global view resolution.
    #[arg(long, help_heading = "Inference")]
    pub base_size: Option<u32>,

    /// Local crop resolution.
    #[arg(long, help_heading = "Inference")]
    pub image_size: Option<u32>,

    /// Enables dynamic crop mode.
    #[arg(long, help_heading = "Inference")]
    pub crop_mode: Option<bool>,

    /// Default max tokens budget per request.
    #[arg(long, help_heading = "Inference")]
    pub max_new_tokens: Option<usize>,

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

    /// Repetition penalty (>1 discourages repeats).
    #[arg(long, help_heading = "Inference")]
    pub repetition_penalty: Option<f32>,

    /// Enforce no-repeat n-gram constraint of the given size.
    #[arg(long, help_heading = "Inference")]
    pub no_repeat_ngram_size: Option<usize>,

    /// RNG seed for sampling.
    #[arg(long, help_heading = "Inference")]
    pub seed: Option<u64>,

    /// Host/IP for Rocket to bind to.
    #[arg(long, help_heading = "Application")]
    pub host: Option<String>,

    /// TCP port for Rocket.
    #[arg(long, help_heading = "Application")]
    pub port: Option<u16>,
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
        overrides.inference.base_size = args.base_size;
        overrides.inference.image_size = args.image_size;
        overrides.inference.crop_mode = args.crop_mode;
        overrides.inference.max_new_tokens = args.max_new_tokens;
        overrides.inference.do_sample = args.do_sample;
        overrides.inference.temperature = args.temperature;
        overrides.inference.top_p = args.top_p;
        overrides.inference.top_k = args.top_k;
        overrides.inference.repetition_penalty = args.repetition_penalty;
        overrides.inference.no_repeat_ngram_size = args.no_repeat_ngram_size;
        overrides.inference.seed = args.seed;
        overrides.server.host = args.host.clone();
        overrides.server.port = args.port;
        overrides
    }
}

impl ConfigOverride for &Args {
    fn apply(self, config: &mut AppConfig) {
        config.apply_overrides(&ConfigOverrides::from(self));
    }
}
