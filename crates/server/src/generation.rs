use std::{convert::TryFrom, sync::Arc};

use base64::Engine;
use deepseek_ocr_pipeline::{
    DecodeParameters, OcrMessage, OcrPrompt, OcrRequest, OcrRole, OcrResponse, VisionSettings,
    deepseek_ocr_core::ocr_inference_engine::{OcrPromptMessage, OcrPromptRole},
};
use image::DynamicImage;
use reqwest::blocking::Client;
use rocket::tokio;
use tokenizers::Tokenizer;
use tracing::{error, info};

use crate::{
    error::ApiError,
    models::{ApiMessage, ImagePayload, MessageContent, MessagePart},
    state::GenerationInputs,
    stream::{StreamContext, StreamController},
};

type StreamCallback = Box<dyn Fn(usize, &[i64])>;

const EMPTY_GENERATION_ERROR: &str =
    "generation failed: model returned empty output (response_tokens=0 and text is empty)";

#[derive(Debug)]
pub struct GenerationResult {
    pub text: String,
    pub rendered_prompt: String,
    pub prompt_tokens: usize,
    pub response_tokens: usize,
}

pub struct ParsedPromptInputs {
    pub messages: Vec<OcrPromptMessage>,
    pub images: Vec<DynamicImage>,
}

struct GenerateBlockingArgs {
    handle: deepseek_ocr_pipeline::OcrPipelineHandle,
    template: String,
    tokenizer: Arc<Tokenizer>,
    messages: Vec<OcrPromptMessage>,
    images: Vec<DynamicImage>,
    vision: VisionSettings,
    params: DecodeParameters,
    stream: Option<StreamContext>,
}

pub async fn generate_async(
    inputs: GenerationInputs,
    messages: Vec<OcrPromptMessage>,
    images: Vec<DynamicImage>,
    params: DecodeParameters,
    stream: Option<StreamContext>,
) -> Result<GenerationResult, ApiError> {
    let stream_for_block = stream.clone();
    let args = GenerateBlockingArgs {
        handle: inputs.handle.clone(),
        template: inputs.template.clone(),
        tokenizer: Arc::clone(&inputs.tokenizer),
        messages,
        images,
        vision: inputs.vision,
        params,
        stream: stream_for_block,
    };
    let join_result = tokio::task::spawn_blocking(move || {
        generate_blocking(args)
    })
    .await;

    match join_result {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(err)) => {
            if stream.is_some() {
                error!(error = %err, "stream generation failed");
            }
            if let Some(ctx) = stream {
                ctx.send_error(&err.to_string());
            }
            Err(err)
        }
        Err(err) => {
            let api_err = ApiError::Internal(format!("generation task failed: {err}"));
            if stream.is_some() {
                error!(error = %api_err, "stream generation task join failed");
            }
            if let Some(ctx) = stream {
                ctx.send_error(&api_err.to_string());
            }
            Err(api_err)
        }
    }
}

fn generate_blocking(args: GenerateBlockingArgs) -> Result<GenerationResult, ApiError> {
    let GenerateBlockingArgs {
        handle,
        template,
        tokenizer,
        messages,
        images,
        vision,
        params,
        stream,
    } = args;

    let tokenizer_ref = tokenizer.as_ref();
    let stream_controller = stream.map(|ctx| StreamController::new(Arc::clone(&tokenizer), ctx));
    let mut callback_box: Option<StreamCallback> = None;
    if let Some(controller) = stream_controller.as_ref() {
        controller.send_initial();
        let callback = controller.callback();
        callback_box = Some(Box::new(callback));
    }

    let request = OcrRequest {
        prompt: OcrPrompt::Messages(convert_prompt_messages(&messages)),
        template,
        system_prompt: String::new(),
        images,
        vision,
        decode: params,
    };

    let decode_result = handle.generate(&request, None, callback_box.as_deref());
    drop(callback_box);

    let outcome = match decode_result {
        Ok(output) => output,
        Err(err) => {
            let message = err.to_string();
            if is_bad_request_generation_error(&message) {
                return Err(ApiError::BadRequest(message));
            }
            return Err(ApiError::Internal(format!("generation failed: {err:#}")));
        }
    };

    let OcrResponse {
        text: normalized,
        rendered_prompt,
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

    if response_tokens == 0 && normalized.trim().is_empty() {
        return Err(ApiError::Internal(EMPTY_GENERATION_ERROR.to_string()));
    }

    if let Some(controller) = stream_controller.as_ref() {
        controller.flush_remaining(&generated_tokens);
        controller.finalize(&normalized, prompt_tokens, response_tokens);
    }

    Ok(GenerationResult {
        text: normalized,
        rendered_prompt,
        prompt_tokens,
        response_tokens,
    })
}

fn convert_prompt_messages(messages: &[OcrPromptMessage]) -> Vec<OcrMessage> {
    messages
        .iter()
        .map(|message| OcrMessage {
            role: match message.role {
                OcrPromptRole::System => OcrRole::System,
                OcrPromptRole::User => OcrRole::User,
                OcrPromptRole::Assistant => OcrRole::Assistant,
            },
            content: message.content.clone(),
        })
        .collect()
}

fn is_bad_request_generation_error(message: &str) -> bool {
    message.contains("prompt formatting failed")
        || message.contains("prompt/image embedding mismatch")
        || message.contains("rendered prompt expects")
}

pub fn collect_prompt_inputs(messages: &[ApiMessage]) -> Result<ParsedPromptInputs, ApiError> {
    let (prompt_messages, images) = collect_prompt_messages(messages)?;
    Ok(ParsedPromptInputs {
        messages: prompt_messages,
        images,
    })
}

fn collect_prompt_messages(
    messages: &[ApiMessage],
) -> Result<(Vec<OcrPromptMessage>, Vec<DynamicImage>), ApiError> {
    let latest_user_idx = messages
        .iter()
        .rposition(|message| message.role.eq_ignore_ascii_case("user"))
        .ok_or_else(|| {
            ApiError::BadRequest("request must include at least one user message".into())
        })?;

    let mut prompt_messages = Vec::new();
    let mut all_images = Vec::new();

    // OCR模型不是为对话训练的，所以只保留一轮的prompt，留多轮连正常输出都产生不了
    for message in &messages[..latest_user_idx] {
        if !message.role.eq_ignore_ascii_case("system") {
            continue;
        }
        let (text, mut msg_images) = flatten_content(&message.content)?;
        if !text.is_empty() {
            prompt_messages.push(OcrPromptMessage::system(text));
        }
        all_images.append(&mut msg_images);
    }

    let (user_text, mut user_images) = flatten_content(&messages[latest_user_idx].content)?;
    if !user_text.is_empty() {
        prompt_messages.push(OcrPromptMessage::user(user_text));
    }
    all_images.append(&mut user_images);

    if prompt_messages.is_empty() && all_images.is_empty() {
        return Err(ApiError::BadRequest(
            "user content must include text or images".into(),
        ));
    }

    Ok((prompt_messages, all_images))
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
