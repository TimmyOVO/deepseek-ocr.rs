use std::{convert::TryFrom, sync::Arc};

use base64::Engine;
use deepseek_ocr_core::{DecodeOutcome, DecodeParameters, ModelKind, VisionSettings};
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

pub fn base_decode_parameters(
    inputs: &GenerationInputs,
    max_new_tokens: usize,
) -> DecodeParameters {
    let mut params = inputs.defaults.clone();
    params.max_new_tokens = max_new_tokens;
    params
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
            inputs.vision,
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
    vision: VisionSettings,
    params: DecodeParameters,
    stream: Option<StreamContext>,
) -> Result<GenerationResult, ApiError> {
    let guard = model
        .lock()
        .map_err(|_| ApiError::Internal("model lock poisoned".into()))?;
    let tokenizer_ref = tokenizer.as_ref();
    let stream_controller = stream.map(|ctx| StreamController::new(Arc::clone(&tokenizer), ctx));
    let mut callback_box: Option<Box<dyn Fn(usize, &[i64])>> = None;
    if let Some(controller) = stream_controller.as_ref() {
        controller.send_initial();
        let callback = controller.callback();
        callback_box = Some(Box::new(callback));
    }

    let decode_result = guard.decode(
        tokenizer_ref,
        &prompt,
        &images,
        vision,
        &params,
        callback_box.as_deref(),
    );
    drop(callback_box);

    let outcome = match decode_result {
        Ok(output) => output,
        Err(err) => {
            drop(guard);
            let message = err.to_string();
            if message.contains("prompt formatting failed")
                || message.contains("prompt/image embedding mismatch")
            {
                return Err(ApiError::BadRequest(message));
            }
            return Err(ApiError::Internal(format!("generation failed: {err:#}")));
        }
    };

    drop(guard);

    let DecodeOutcome {
        text: normalized,
        prompt_tokens,
        response_tokens,
        generated_tokens,
    } = outcome;

    let decoded = tokenizer_ref
        .decode(
            &generated_tokens
                .iter()
                .filter_map(|&id| u32::try_from(id).ok())
                .collect::<Vec<_>>(),
            true,
        )
        .unwrap_or_default();

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

    if let Some(controller) = stream_controller.as_ref() {
        controller.flush_remaining(&generated_tokens);
        controller.finalize(&normalized, prompt_tokens, response_tokens);
    }

    Ok(GenerationResult {
        text: normalized,
        prompt_tokens,
        response_tokens,
    })
}

pub fn convert_messages(
    kind: ModelKind,
    messages: &[ApiMessage],
) -> Result<(String, Vec<DynamicImage>), ApiError> {
    match kind {
        ModelKind::Deepseek => convert_deepseek_messages(messages),
        ModelKind::PaddleOcrVl | ModelKind::DotsOcr => convert_paddle_messages(messages),
    }
}

fn convert_deepseek_messages(
    messages: &[ApiMessage],
) -> Result<(String, Vec<DynamicImage>), ApiError> {
    let (sections, images) = collect_prompt_sections(messages)?;
    let mut prompt = String::from("");
    let body = sections.join("\n\n").trim().to_owned();
    prompt.push_str(&body);

    Ok((prompt, images))
}

fn convert_paddle_messages(
    messages: &[ApiMessage],
) -> Result<(String, Vec<DynamicImage>), ApiError> {
    let (sections, images) = collect_prompt_sections(messages)?;
    let prompt = sections.join("\n\n").trim().to_owned();
    Ok((prompt, images))
}

fn collect_prompt_sections(
    messages: &[ApiMessage],
) -> Result<(Vec<String>, Vec<DynamicImage>), ApiError> {
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
        if !message.role.eq_ignore_ascii_case("system") {
            continue;
        }
        let (text, mut msg_images) = flatten_content(&message.content)?;
        if !text.is_empty() {
            sections.push(text);
        }
        all_images.append(&mut msg_images);
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

    if sections.is_empty() && all_images.is_empty() {
        return Err(ApiError::BadRequest(
            "user content must include text or images".into(),
        ));
    }

    Ok((sections, all_images))
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
