use std::{
    env, fmt,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use candle_core::Device;
use once_cell::sync::Lazy;
use tracing::{info, warn};

/// Supported runtime quantization algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationKind {
    None,
    Q8_0,
    Q4K,
}

impl QuantizationKind {
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl fmt::Display for QuantizationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Q8_0 => f.write_str("Q8_0"),
            Self::Q4K => f.write_str("Q4_K"),
        }
    }
}

/// High-level components that may opt into quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearLayerGroup {
    Text,
    Projector,
    Vision,
}

/// Which modules should participate in quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationTargets {
    Text,
    TextAndProjector,
}

impl fmt::Display for QuantizationTargets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => f.write_str("text"),
            Self::TextAndProjector => f.write_str("text+projector"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizationConfig {
    pub kind: QuantizationKind,
    pub targets: QuantizationTargets,
    pub keep_full_precision_weights: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QuantizationStats {
    pub candidates: usize,
    pub quantized: usize,
    pub fallback: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum QuantizationOutcome {
    Quantized,
    Fallback,
}

pub struct QuantizationState {
    config: QuantizationConfig,
    stats: Mutex<QuantizationStats>,
    summary_logged: AtomicBool,
}

impl QuantizationState {
    fn from_env() -> Self {
        let kind = parse_kind_from_env();
        let targets = parse_targets_from_env();
        let keep_full_precision_weights = parse_keep_full_precision_from_env();
        Self {
            config: QuantizationConfig {
                kind,
                targets,
                keep_full_precision_weights,
            },
            stats: Mutex::new(QuantizationStats::default()),
            summary_logged: AtomicBool::new(false),
        }
    }

    pub fn global() -> &'static Self {
        static INSTANCE: Lazy<QuantizationState> = Lazy::new(QuantizationState::from_env);
        &INSTANCE
    }

    pub fn config(&self) -> QuantizationConfig {
        self.config
    }

    pub fn enabled_for(&self, group: LinearLayerGroup) -> bool {
        if !self.config.kind.is_enabled() {
            return false;
        }
        match group {
            LinearLayerGroup::Text => true,
            LinearLayerGroup::Projector => {
                matches!(self.config.targets, QuantizationTargets::TextAndProjector)
            }
            LinearLayerGroup::Vision => false,
        }
    }

    pub fn record_attempt(&self, outcome: QuantizationOutcome) {
        let mut stats = self
            .stats
            .lock()
            .expect("quantization stats mutex poisoned");
        stats.candidates += 1;
        match outcome {
            QuantizationOutcome::Quantized => stats.quantized += 1,
            QuantizationOutcome::Fallback => stats.fallback += 1,
        }
    }

    pub fn log_summary(&self, device: &Device) {
        if self.summary_logged.swap(true, Ordering::SeqCst) {
            return;
        }
        let stats = self
            .stats
            .lock()
            .expect("quantization stats mutex poisoned")
            .clone();
        info!(
            backend = backend_label(device),
            quant = %self.config.kind,
            targets = %self.config.targets,
            keep_fp = self.config.keep_full_precision_weights,
            candidates = stats.candidates,
            quantized = stats.quantized,
            fallback = stats.fallback,
            "language runtime quantization summary"
        );
    }
}

fn parse_kind_from_env() -> QuantizationKind {
    match env::var("DEEPSEEK_OCR_QUANT") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "" | "none" => QuantizationKind::None,
            "q8_0" | "q8" | "q8.0" => QuantizationKind::Q8_0,
            "q4_k" | "q4k" => {
                warn!("DEEPSEEK_OCR_QUANT=Q4_K not yet implemented, falling back to float weights");
                QuantizationKind::Q4K
            }
            other => {
                warn!(
                    "unsupported DEEPSEEK_OCR_QUANT value `{other}`, expected none|Q8_0|Q4_K; disabling quantization"
                );
                QuantizationKind::None
            }
        },
        Err(_) => QuantizationKind::None,
    }
}

fn parse_targets_from_env() -> QuantizationTargets {
    match env::var("DEEPSEEK_OCR_QUANT_TARGETS") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "" | "text" => QuantizationTargets::Text,
            "text+projector" | "text,projector" | "projector+text" => {
                QuantizationTargets::TextAndProjector
            }
            other => {
                warn!("unsupported DEEPSEEK_OCR_QUANT_TARGETS value `{other}`, defaulting to text");
                QuantizationTargets::Text
            }
        },
        Err(_) => QuantizationTargets::Text,
    }
}

fn parse_keep_full_precision_from_env() -> bool {
    match env::var("DEEPSEEK_OCR_QUANT_KEEP_FLOAT") {
        Ok(value) => matches!(value.trim(), "" | "1" | "true" | "TRUE"),
        Err(_) => false,
    }
}

fn backend_label(device: &Device) -> &'static str {
    if device.is_cuda() {
        "CUDA"
    } else if device.is_metal() {
        "Metal"
    } else {
        "CPU"
    }
}
