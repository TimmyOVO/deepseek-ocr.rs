use std::{
    env, fmt,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, quantized::QMatMul};
use once_cell::sync::Lazy;
use tracing::info;

/// High-level components that may opt into quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearLayerGroup {
    Text,
    Projector,
    Vision,
}

/// Module categories used for per-module stats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantModule {
    TextLinear,
    Projector,
    LmHead,
}

impl fmt::Display for QuantModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TextLinear => f.write_str("text-linear"),
            Self::Projector => f.write_str("projector"),
            Self::LmHead => f.write_str("lm_head"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizationConfig {
    pub verbose_per_layer: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ModuleStats {
    pub candidates: usize,
    pub quantized: usize,
    pub fallback: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QuantizationStats {
    pub candidates: usize,
    pub quantized: usize,
    pub fallback: usize,
    pub text_linear: ModuleStats,
    pub projector: ModuleStats,
    pub lm_head: ModuleStats,
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
        let verbose_per_layer = parse_verbose_from_env();
        Self {
            config: QuantizationConfig { verbose_per_layer },
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

    pub fn verbose(&self) -> bool {
        self.config.verbose_per_layer
    }

    pub fn record_attempt(&self, module: QuantModule, outcome: QuantizationOutcome) {
        let mut stats = self
            .stats
            .lock()
            .expect("quantization stats mutex poisoned");
        stats.candidates += 1;
        match outcome {
            QuantizationOutcome::Quantized => stats.quantized += 1,
            QuantizationOutcome::Fallback => stats.fallback += 1,
        }
        let mstats = match module {
            QuantModule::TextLinear => &mut stats.text_linear,
            QuantModule::Projector => &mut stats.projector,
            QuantModule::LmHead => &mut stats.lm_head,
        };
        mstats.candidates += 1;
        match outcome {
            QuantizationOutcome::Quantized => mstats.quantized += 1,
            QuantizationOutcome::Fallback => mstats.fallback += 1,
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
            verbose = self.config.verbose_per_layer,
            candidates = stats.candidates,
            quantized = stats.quantized,
            fallback = stats.fallback,
            text_cand = stats.text_linear.candidates,
            text_quant = stats.text_linear.quantized,
            text_fallback = stats.text_linear.fallback,
            proj_cand = stats.projector.candidates,
            proj_quant = stats.projector.quantized,
            proj_fallback = stats.projector.fallback,
            lm_cand = stats.lm_head.candidates,
            lm_quant = stats.lm_head.quantized,
            lm_fallback = stats.lm_head.fallback,
            "quantization summary"
        );
    }
}

fn parse_verbose_from_env() -> bool {
    match env::var("DEEPSEEK_OCR_QUANT_VERBOSE") {
        Ok(value) => matches!(value.trim(), "" | "1" | "true" | "TRUE"),
        Err(_) => false,
    }
}

pub(crate) fn backend_label(device: &Device) -> &'static str {
    if device.is_cuda() {
        "CUDA"
    } else if device.is_metal() {
        "Metal"
    } else {
        "CPU"
    }
}

pub fn run_quantized_matmul(_label: &str, qm: &QMatMul, input: &Tensor) -> Result<Tensor> {
    let dtype = input.dtype();
    let device = input.device();
    if device.is_cuda() || device.is_metal() {
        let mut out = if dtype == DType::F32 {
            qm.forward(input)?
        } else {
            let activations = input.to_dtype(DType::F32)?;
            qm.forward(&activations)?
        };
        if out.dtype() != dtype {
            out = out.to_dtype(dtype)?;
        }
        Ok(out)
    } else {
        let mut out = qm.forward(input)?;
        if out.dtype() != dtype {
            out = out.to_dtype(dtype)?;
        }
        Ok(out)
    }
}
