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

    /// Host/IP for Rocket to bind to.
    #[arg(long, help_heading = "Application")]
    pub host: Option<String>,

    /// TCP port for Rocket.
    #[arg(long, help_heading = "Application")]
    pub port: Option<u16>,

    /// Model identifier returned by /models.
    #[arg(long, help_heading = "Application")]
    pub model_id: Option<String>,
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
        overrides.server.host = args.host.clone();
        overrides.server.port = args.port;
        overrides.server.model_id = args.model_id.clone();
        overrides
    }
}

impl ConfigOverride for &Args {
    fn apply(self, config: &mut AppConfig) {
        config.apply_overrides(&ConfigOverrides::from(self));
    }
}
