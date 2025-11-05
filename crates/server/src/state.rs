use std::sync::{Arc, Mutex};

use tokenizers::Tokenizer;

use deepseek_ocr_core::model::DeepseekOcrModel;

pub type SharedModel = Arc<Mutex<DeepseekOcrModel>>;

pub struct AppState {
    pub model: SharedModel,
    pub tokenizer: Arc<Tokenizer>,
    pub base_size: u32,
    pub image_size: u32,
    pub crop_mode: bool,
    pub max_new_tokens: usize,
    pub use_cache: bool,
    pub do_sample: bool,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: Option<usize>,
    pub seed: Option<u64>,
    pub model_id: String,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: SharedModel,
        tokenizer: Arc<Tokenizer>,
        base_size: u32,
        image_size: u32,
        crop_mode: bool,
        max_new_tokens: usize,
        use_cache: bool,
        do_sample: bool,
        temperature: f64,
        top_p: f64,
        top_k: Option<usize>,
        repetition_penalty: f32,
        no_repeat_ngram_size: Option<usize>,
        seed: Option<u64>,
        model_id: String,
    ) -> Self {
        Self {
            model,
            tokenizer,
            base_size,
            image_size,
            crop_mode,
            max_new_tokens,
            use_cache,
            do_sample,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            no_repeat_ngram_size,
            seed,
            model_id,
        }
    }
}

#[derive(Clone)]
pub struct GenerationInputs {
    pub model: SharedModel,
    pub tokenizer: Arc<Tokenizer>,
    pub base_size: u32,
    pub image_size: u32,
    pub crop_mode: bool,
    pub use_cache: bool,
    pub do_sample: bool,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: Option<usize>,
    pub seed: Option<u64>,
}

impl GenerationInputs {
    pub fn from_app(state: &AppState) -> Self {
        Self {
            model: Arc::clone(&state.model),
            tokenizer: Arc::clone(&state.tokenizer),
            base_size: state.base_size,
            image_size: state.image_size,
            crop_mode: state.crop_mode,
            use_cache: state.use_cache,
            do_sample: state.do_sample,
            temperature: state.temperature,
            top_p: state.top_p,
            top_k: state.top_k,
            repetition_penalty: state.repetition_penalty,
            no_repeat_ngram_size: state.no_repeat_ngram_size,
            seed: state.seed,
        }
    }
}
