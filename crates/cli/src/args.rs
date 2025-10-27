use std::path::PathBuf;

use clap::Parser;
use deepseek_ocr_config::{AppConfig, ConfigOverride, ConfigOverrides};
use deepseek_ocr_core::runtime::{DeviceKind, Precision};

#[derive(Parser, Debug)]
#[command(author, version, about = "DeepSeek-OCR CLI", long_about = None)]
pub struct Args {
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
