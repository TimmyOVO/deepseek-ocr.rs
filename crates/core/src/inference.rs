use anyhow::{Context, Result};
use candle_core::Device;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::{benchmark::Timer, conversation::get_conv_template, sampling::TokenSelectionParams};

/// Callback used to stream decoded token pieces.
pub type StreamCallback<'a> = Option<&'a dyn Fn(usize, &[i64])>;

/// Vision pre-processing knobs shared across OCR backends.
#[derive(Debug, Clone, Copy)]
pub struct VisionSettings {
    pub base_size: u32,
    pub image_size: u32,
    pub crop_mode: bool,
}

/// Decoding parameters that map directly onto generation options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DecodeParameters {
    pub max_new_tokens: usize,
    pub do_sample: bool,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: Option<usize>,
    pub seed: Option<u64>,
    pub use_cache: bool,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct DecodeParametersPatch {
    pub max_new_tokens: Option<usize>,
    pub do_sample: Option<bool>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub no_repeat_ngram_size: Option<usize>,
    pub seed: Option<u64>,
    pub use_cache: Option<bool>,
}

impl DecodeParameters {
    pub fn with_sampling_defaults(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            do_sample: false,
            temperature: 0.0,
            top_p: None,
            top_k: None,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: None,
            seed: None,
            use_cache: true,
        }
    }
}

impl Default for DecodeParameters {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            do_sample: false,
            temperature: 0.0,
            top_p: Some(1.0),
            top_k: None,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: Some(20),
            seed: None,
            use_cache: true,
        }
    }
}

impl std::ops::AddAssign<&DecodeParametersPatch> for DecodeParameters {
    fn add_assign(&mut self, rhs: &DecodeParametersPatch) {
        if let Some(max_new_tokens) = rhs.max_new_tokens {
            self.max_new_tokens = max_new_tokens;
        }
        if let Some(sample) = rhs.do_sample {
            self.do_sample = sample;
        }
        if let Some(temp) = rhs.temperature {
            self.temperature = temp;
        }
        if let Some(prob) = rhs.top_p {
            self.top_p = normalize_top_p(prob);
        }
        if let Some(top_k) = rhs.top_k {
            self.top_k = normalize_top_k(top_k);
        }
        if let Some(repetition_penalty) = rhs.repetition_penalty {
            self.repetition_penalty = repetition_penalty;
        }
        if let Some(no_repeat) = rhs.no_repeat_ngram_size {
            self.no_repeat_ngram_size = normalize_no_repeat_ngram_size(no_repeat);
        }
        if let Some(seed) = rhs.seed {
            self.seed = Some(seed);
        }
        if let Some(use_cache) = rhs.use_cache {
            self.use_cache = use_cache;
        }
    }
}

impl std::ops::Add<&DecodeParametersPatch> for DecodeParameters {
    type Output = DecodeParameters;

    fn add(mut self, rhs: &DecodeParametersPatch) -> Self::Output {
        self += rhs;
        self
    }
}

fn normalize_top_p(probability: f64) -> Option<f64> {
    (probability < 1.0).then_some(probability)
}

fn normalize_top_k(top_k: usize) -> Option<usize> {
    (top_k > 0).then_some(top_k)
}

fn normalize_no_repeat_ngram_size(size: usize) -> Option<usize> {
    (size > 0).then_some(size)
}

impl TokenSelectionParams for DecodeParameters {
    fn do_sample(&self) -> bool {
        self.do_sample
    }

    fn temperature(&self) -> f64 {
        self.temperature
    }

    fn top_p(&self) -> Option<f64> {
        self.top_p
    }

    fn top_k(&self) -> Option<usize> {
        self.top_k
    }

    fn repetition_penalty(&self) -> f32 {
        self.repetition_penalty
    }

    fn no_repeat_ngram_size(&self) -> Option<usize> {
        self.no_repeat_ngram_size
    }
}

/// Collected results from a decode call.
#[derive(Debug)]
pub struct DecodeOutcome {
    pub text: String,
    pub prompt_tokens: usize,
    pub response_tokens: usize,
    pub generated_tokens: Vec<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Deepseek,
    PaddleOcrVl,
    DotsOcr,
    GlmOcr,
}

#[derive(Debug)]
pub struct ModelLoadArgs<'a> {
    pub kind: ModelKind,
    pub config_path: Option<&'a std::path::Path>,
    pub weights_path: Option<&'a std::path::Path>,
    pub snapshot_path: Option<&'a std::path::Path>,
    pub device: Device,
    pub dtype: candle_core::DType,
}

/// Shared interface implemented by all OCR inference backends.
pub trait OcrEngine: Send {
    fn kind(&self) -> ModelKind;
    fn device(&self) -> &Device;
    fn dtype(&self) -> candle_core::DType;
    fn weights_path(&self) -> Option<&std::path::Path> {
        None
    }
    fn flash_attention_enabled(&self) -> bool {
        false
    }

    fn decode(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        images: &[DynamicImage],
        vision: VisionSettings,
        params: &DecodeParameters,
        stream: StreamCallback,
    ) -> Result<DecodeOutcome>;
}

/// Render a prompt using the configured conversation template and system prompt.
pub fn render_prompt(template: &str, system_prompt: &str, raw_prompt: &str) -> Result<String> {
    let timer = Timer::new("prompt.render");
    let mut template = get_conv_template(template)
        .with_context(|| format!("unknown conversation template {template}"))?;
    template.set_system_message(system_prompt.to_owned());
    template.reset_messages();
    template.append_message("User", Some(raw_prompt.to_owned()));
    template.append_message("Assistant", None);
    let prompt = template.get_prompt();
    timer.finish(|event| {
        event.add_field("chars", prompt.len() as u64);
    });
    Ok(prompt)
}

/// Normalise decoder output by stripping sentinel tokens and Windows line-endings.
pub fn normalize_text(s: &str) -> String {
    s.replace("\r\n", "\n")
        .replace("<｜end▁of▁sentence｜>", "")
        .trim()
        .to_string()
}
