use std::path::PathBuf;

use clap::Args;
use deepseek_ocr_core::runtime::{DeviceKind, Precision};

use crate::config::{ConfigOverrides, ServerOverride};

#[derive(Args, Debug, Clone, Default)]
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

#[derive(Args, Debug, Clone, Default)]
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

#[derive(Args, Debug, Clone, Default)]
pub struct ServerBindArgs {
    /// Host/IP for server to bind.
    #[arg(long, help_heading = "Application")]
    pub host: Option<String>,

    /// TCP port for server.
    #[arg(long, help_heading = "Application")]
    pub port: Option<u16>,
}

impl From<&CommonInferenceArgs> for crate::InferenceOverride {
    fn from(value: &CommonInferenceArgs) -> Self {
        Self {
            device: value.device,
            precision: value.dtype,
            template: value.template.clone(),
            base_size: value.base_size,
            image_size: value.image_size,
            crop_mode: value.crop_mode,
            decode: deepseek_ocr_core::DecodeParametersPatch {
                max_new_tokens: value.max_new_tokens,
                do_sample: value.do_sample,
                temperature: value.temperature,
                top_p: value.top_p,
                top_k: value.top_k,
                repetition_penalty: value.repetition_penalty,
                no_repeat_ngram_size: value.no_repeat_ngram_size,
                seed: value.seed,
                use_cache: value.no_cache.then_some(false),
            },
        }
    }
}

pub fn build_config_overrides(
    model: &CommonModelArgs,
    inference: &CommonInferenceArgs,
    bind: Option<&ServerBindArgs>,
) -> ConfigOverrides {
    ConfigOverrides {
        config_path: model.config.clone(),
        model_id: model.model.clone(),
        model_config: model.model_config.clone(),
        tokenizer: model.tokenizer.clone(),
        weights: model.weights.clone(),
        inference: inference.into(),
        server: ServerOverride {
            host: bind.and_then(|value| value.host.clone()),
            port: bind.and_then(|value| value.port),
        },
    }
}
