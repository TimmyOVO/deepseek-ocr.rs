use std::{convert::TryFrom, sync::Arc};

use base64::Engine;
use candle_core::{DType, Tensor};
use deepseek_ocr_core::{
    inference::{
        build_prompt_tokens, compute_image_embeddings, normalize_text, prepare_vision_inputs,
    },
    model::{DeepseekOcrModel, GenerateOptions, OwnedVisionInput},
};
use image::DynamicImage;
use reqwest::blocking::Client;
use rocket::tokio;
use tokenizers::Tokenizer;
use tracing::info;

use crate::{
    error::ApiError,
    models::{ApiMessage, ImagePayload, MessageContent, MessagePart},
    state::{GenerationInputs, SharedModel},
    stream::{StreamContext, StreamController},
};

#[derive(Debug)]
pub struct GenerationResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub response_tokens: usize,
}

#[derive(Debug, Clone)]
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

impl DecodeParameters {
    pub fn from_inputs(inputs: &GenerationInputs, max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            do_sample: inputs.do_sample,
            temperature: inputs.temperature,
            top_p: if inputs.top_p < 1.0 {
                Some(inputs.top_p)
            } else {
                None
            },
            top_k: inputs.top_k,
            repetition_penalty: inputs.repetition_penalty,
            no_repeat_ngram_size: inputs.no_repeat_ngram_size,
            seed: inputs.seed,
            use_cache: inputs.use_cache,
        }
    }
}

pub async fn generate_async(
    inputs: GenerationInputs,
    prompt: String,
    images: Vec<DynamicImage>,
    params: DecodeParameters,
    stream: Option<StreamContext>,
) -> Result<GenerationResult, ApiError> {
    let stream_for_block = stream.clone();
    let join_result = tokio::task::spawn_blocking(move || {
        generate_blocking(
            &inputs.model,
            Arc::clone(&inputs.tokenizer),
            prompt,
            images,
            inputs.base_size,
            inputs.image_size,
            inputs.crop_mode,
            params,
            stream_for_block,
        )
    })
    .await;

    match join_result {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(err)) => {
            if let Some(ctx) = stream {
                ctx.send_error(&err.to_string());
            }
            Err(err)
        }
        Err(err) => {
            let api_err = ApiError::Internal(format!("generation task failed: {err}"));
            if let Some(ctx) = stream {
                ctx.send_error(&api_err.to_string());
            }
            Err(api_err)
        }
    }
}

fn generate_blocking(
    model: &SharedModel,
    tokenizer: Arc<Tokenizer>,
    prompt: String,
    images: Vec<DynamicImage>,
    base_size: u32,
    image_size: u32,
    crop_mode: bool,
    params: DecodeParameters,
    stream: Option<StreamContext>,
) -> Result<GenerationResult, ApiError> {
    let guard = model
        .lock()
        .map_err(|_| ApiError::Internal("model lock poisoned".into()))?;
    let tokenizer_ref = tokenizer.as_ref();
    let stream_controller = stream.map(|ctx| StreamController::new(Arc::clone(&tokenizer), ctx));
    let owned_inputs = prepare_inputs(&*guard, &images, base_size, image_size, crop_mode)?;
    let embeddings = compute_image_embeddings(&*guard, &owned_inputs)
        .map_err(|err| ApiError::Internal(format!("image embedding failed: {err:#}")))?;
    let (input_ids_vec, mask_vec) = build_prompt_tokens(
        tokenizer_ref,
        &prompt,
        &embeddings,
        &owned_inputs,
        base_size,
        image_size,
        crop_mode,
    )
    .map_err(|err| ApiError::BadRequest(format!("prompt formatting failed: {err:#}")))?;

    let input_len = input_ids_vec.len();
    let token_device = guard.device();

    let input_ids = Tensor::from_vec(input_ids_vec.clone(), (1, input_len), token_device)
        .map_err(|err| ApiError::Internal(format!("input tensor failed: {err}")))?
        .to_dtype(DType::I64)
        .map_err(|err| ApiError::Internal(format!("tensor cast failed: {err}")))?;

    let mask_tensor = Tensor::from_vec(mask_vec.clone(), (1, mask_vec.len()), token_device)
        .map_err(|err| ApiError::Internal(format!("mask tensor failed: {err}")))?
        .to_dtype(DType::U8)
        .map_err(|err| ApiError::Internal(format!("mask cast failed: {err}")))?;

    let mut options = GenerateOptions::new(params.max_new_tokens);
    options.images_seq_mask = Some(&mask_tensor);
    if !embeddings.is_empty() {
        options.image_embeddings = Some(embeddings.as_slice());
    }
    options.eos_token_id = guard.language_model().config().eos_token_id;
    options.use_cache = params.use_cache;
    options.do_sample = params.do_sample;
    options.temperature = params.temperature;
    options.top_p = params.top_p;
    options.top_k = params.top_k;
    options.repetition_penalty = params.repetition_penalty;
    options.no_repeat_ngram_size = params.no_repeat_ngram_size;
    options.seed = params.seed;

    let mut _progress_guard: Option<Box<dyn Fn(usize, &[i64]) + Send + Sync>> = None;
    if let Some(controller) = &stream_controller {
        controller.send_initial();
        let callback = controller.callback();
        _progress_guard = Some(Box::new(callback));
        if let Some(cb) = _progress_guard.as_ref() {
            options.progress_callback = Some(&**cb);
        }
    }

    let generated = guard
        .generate(&input_ids, options)
        .map_err(|err| ApiError::Internal(format!("generation failed: {err:#}")))?;
    let generated_tokens = generated
        .to_vec2::<i64>()
        .map_err(|err| ApiError::Internal(format!("token decode failed: {err:#}")))?
        .into_iter()
        .next()
        .unwrap_or_default();
    let decoded = tokenizer_ref
        .decode(
            &generated_tokens
                .iter()
                .filter_map(|&id| u32::try_from(id).ok())
                .collect::<Vec<_>>(),
            true,
        )
        .unwrap_or_default();
    let normalized = normalize_text(&decoded);

    info!(
        "[generate] decoded_raw=\"{}\" normalized=\"{}\"",
        decoded
            .replace('\n', "\\n")
            .chars()
            .take(120)
            .collect::<String>(),
        normalized
            .replace('\n', "\\n")
            .chars()
            .take(120)
            .collect::<String>()
    );

    drop(guard);

    if let Some(controller) = &stream_controller {
        controller.flush_remaining(&generated_tokens);
        controller.finalize(&normalized, input_len, generated_tokens.len());
    }

    Ok(GenerationResult {
        text: normalized,
        prompt_tokens: input_len,
        response_tokens: generated_tokens.len(),
    })
}

fn prepare_inputs(
    model: &DeepseekOcrModel,
    images: &[DynamicImage],
    base_size: u32,
    image_size: u32,
    crop_mode: bool,
) -> Result<Vec<OwnedVisionInput>, ApiError> {
    prepare_vision_inputs(model, images, base_size, image_size, crop_mode)
        .map_err(|err| ApiError::Internal(format!("vision input failed: {err:#}")))
}

pub fn convert_messages(messages: &[ApiMessage]) -> Result<(String, Vec<DynamicImage>), ApiError> {
    let latest_user_idx = messages
        .iter()
        .rposition(|message| message.role.eq_ignore_ascii_case("user"))
        .ok_or_else(|| {
            ApiError::BadRequest("request must include at least one user message".into())
        })?;

    let mut sections = Vec::new();
    let mut all_images = Vec::new();

    // OCR模型不是为对话训练的，所以只保留一轮的prompt，留多轮连正常输出都产生不了
    for message in &messages[..latest_user_idx] {
        if message.role.eq_ignore_ascii_case("system") {
            let (text, mut msg_images) = flatten_content(&message.content)?;
            if !text.is_empty() {
                sections.push(text);
            }
            all_images.append(&mut msg_images);
        }
    }

    let (user_text, mut user_images) = flatten_content(&messages[latest_user_idx].content)?;
    if !user_text.is_empty() {
        sections.push(user_text);
    }
    all_images.append(&mut user_images);

    if sections.is_empty() && all_images.is_empty() {
        return Err(ApiError::BadRequest(
            "user content must include text or images".into(),
        ));
    }

    let mut prompt = String::from("<|User|>\n");
    let body = sections.join("\n\n");
    if !body.is_empty() {
        prompt.push_str(&body);
        if !body.ends_with('\n') {
            prompt.push('\n');
        }
    }
    prompt.push_str("<|Assistant|>\n");
    Ok((prompt, all_images))
}

fn flatten_content(content: &MessageContent) -> Result<(String, Vec<DynamicImage>), ApiError> {
    match content {
        MessageContent::Text(text) => Ok((text.trim().to_owned(), Vec::new())),
        MessageContent::Parts(parts) => {
            let mut buffer = String::new();
            let mut images = Vec::new();
            for part in parts.iter().rev() {
                match part {
                    MessagePart::ImageUrl { image_url } | MessagePart::InputImage { image_url } => {
                        buffer.push_str("<image>");
                        images.push(load_image(image_url)?);
                    }
                    MessagePart::Text { text } | MessagePart::InputText { text } => {
                        if !buffer.is_empty() {
                            buffer.push('\n');
                        }
                        buffer.push_str(text);
                    }
                }
            }
            Ok((buffer.trim().to_owned(), images))
        }
    }
}

fn load_image(spec: &ImagePayload) -> Result<DynamicImage, ApiError> {
    let url = spec.url();
    if let Some(rest) = url.strip_prefix("data:") {
        return load_data_url(rest);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return fetch_remote_image(url);
    }
    Err(ApiError::BadRequest(
        "only data: URIs or http(s) image URLs are supported".into(),
    ))
}

fn load_data_url(data: &str) -> Result<DynamicImage, ApiError> {
    let (meta, payload) = data
        .split_once(',')
        .ok_or_else(|| ApiError::BadRequest("invalid data URL".into()))?;
    if !meta.ends_with(";base64") {
        return Err(ApiError::BadRequest(
            "data URLs must specify base64 encoding".into(),
        ));
    }
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|err| ApiError::BadRequest(format!("invalid base64 image payload: {err}")))?;
    image::load_from_memory(&decoded)
        .map_err(|err| ApiError::BadRequest(format!("failed to decode inline image: {err}")))
}

fn fetch_remote_image(url: &str) -> Result<DynamicImage, ApiError> {
    let client = Client::new();
    let response = client
        .get(url)
        .send()
        .map_err(|err| ApiError::BadRequest(format!("failed to fetch {url}: {err}")))?
        .error_for_status()
        .map_err(|err| ApiError::BadRequest(format!("image request failed for {url}: {err}")))?;
    let bytes = response
        .bytes()
        .map_err(|err| ApiError::BadRequest(format!("failed to read image body: {err}")))?;
    image::load_from_memory(&bytes)
        .map_err(|err| ApiError::BadRequest(format!("failed to decode remote image: {err}")))
}
