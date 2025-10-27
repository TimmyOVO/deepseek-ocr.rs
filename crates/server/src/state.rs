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
        model_id: String,
    ) -> Self {
        Self {
            model,
            tokenizer,
            base_size,
            image_size,
            crop_mode,
            max_new_tokens,
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
}

impl GenerationInputs {
    pub fn from_app(state: &AppState) -> Self {
        Self {
            model: Arc::clone(&state.model),
            tokenizer: Arc::clone(&state.tokenizer),
            base_size: state.base_size,
            image_size: state.image_size,
            crop_mode: state.crop_mode,
        }
    }
}
