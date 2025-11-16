use std::{sync::Arc, time::SystemTime};

use rocket::{Either, Route, State, serde::json::Json, tokio::sync::mpsc};
use tracing::debug;
use uuid::Uuid;

use deepseek_ocr_core::{DecodeParameters, ModelKind};

use crate::{
    error::ApiError,
    generation::{base_decode_parameters, convert_messages, generate_async},
    models::{
        ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessageResponse, ModelInfo,
        ModelsResponse, ResponseContent, ResponseOutput, ResponsesRequest, ResponsesResponse,
        Usage,
    },
    state::{AppState, GenerationInputs},
    stream::{BoxEventStream, StreamContext, StreamController, StreamKind, into_event_stream},
};

#[get("/health")]
pub fn health() -> &'static str {
    "ok"
}

#[get("/models")]
pub fn list_models(state: &State<AppState>) -> Json<ModelsResponse> {
    let now = current_timestamp();
    let data = state
        .available_models()
        .iter()
        .map(|entry| ModelInfo {
            id: entry.id.clone(),
            object: "model".into(),
            created: now,
            owned_by: match entry.kind {
                ModelKind::Deepseek => "deepseek-ocr".into(),
                ModelKind::PaddleOcrVl => "paddleocr-vl".into(),
                ModelKind::DotsOcr => "dots-ocr".into(),
            },
        })
        .collect();
    Json(ModelsResponse {
        object: "list".into(),
        data,
    })
}

#[options("/models")]
pub fn options_models() -> rocket::http::Status {
    rocket::http::Status::Ok
}

#[post("/responses", format = "json", data = "<req>")]
pub async fn responses_endpoint(
    state: &State<AppState>,
    req: Json<ResponsesRequest>,
) -> Result<Either<Json<ResponsesResponse>, BoxEventStream>, ApiError> {
    let (gen_inputs, active_model_id) = state.prepare_generation(&req.model)?;
    let (prompt, images) = convert_messages(gen_inputs.kind, &req.input)?;
    if prompt_missing_image(&prompt) {
        let fallback = missing_image_markdown();
        if req.stream.unwrap_or(false) {
            return Ok(Either::Right(stream_fallback_response(
                fallback,
                &gen_inputs,
                active_model_id,
            )));
        }
        let response = fallback_response_response(active_model_id, &fallback);
        return Ok(Either::Left(Json(response)));
    }
    let max_tokens = req
        .max_output_tokens
        .or(req.max_tokens)
        .unwrap_or(state.default_max_new_tokens());
    let mut decode = base_decode_parameters(&gen_inputs, max_tokens);
    apply_decode_overrides(
        &mut decode,
        req.do_sample,
        req.temperature,
        req.top_p,
        req.top_k,
        req.repetition_penalty,
        req.no_repeat_ngram_size,
        req.seed,
        req.use_cache,
    );
    if req.stream.unwrap_or(false) {
        let stream_inputs = gen_inputs.clone();
        let decode_for_task = decode.clone();
        let created = current_timestamp();
        let response_id = format!("resp-{}", Uuid::new_v4());
        let output_id = format!("msg-{}", Uuid::new_v4());
        let (sender, rx) = mpsc::unbounded_channel();
        let stream = into_event_stream(rx);
        let context = StreamContext {
            sender,
            kind: StreamKind::Responses {
                response_id: response_id.clone(),
                output_id: output_id.clone(),
                model: active_model_id.clone(),
                created,
            },
        };
        let task_context = context.clone();
        rocket::tokio::spawn(async move {
            let _ = generate_async(
                stream_inputs,
                prompt,
                images,
                decode_for_task,
                Some(task_context),
            )
            .await;
        });
        return Ok(Either::Right(stream));
    }
    let generation = generate_async(gen_inputs, prompt, images, decode, None).await?;
    let created = current_timestamp();
    let response = ResponsesResponse {
        id: format!("resp-{}", Uuid::new_v4()),
        object: "response".into(),
        created,
        model: active_model_id,
        output: vec![ResponseOutput {
            id: format!("msg-{}", Uuid::new_v4()),
            r#type: "message".into(),
            role: "assistant".into(),
            content: vec![ResponseContent {
                r#type: "output_text".into(),
                text: generation.text.clone(),
            }],
        }],
        usage: Usage {
            prompt_tokens: generation.prompt_tokens,
            completion_tokens: generation.response_tokens,
            total_tokens: generation.prompt_tokens + generation.response_tokens,
        },
    };
    Ok(Either::Left(Json(response)))
}

#[post("/chat/completions", format = "json", data = "<req>")]
pub async fn chat_completions_endpoint(
    state: &State<AppState>,
    req: Json<ChatCompletionRequest>,
) -> Result<Either<Json<ChatCompletionResponse>, BoxEventStream>, ApiError> {
    let (gen_inputs, active_model_id) = state.prepare_generation(&req.model)?;
    let (prompt, images) = convert_messages(gen_inputs.kind, &req.messages)?;
    if prompt_missing_image(&prompt) {
        let fallback = missing_image_markdown();
        if req.stream.unwrap_or(false) {
            return Ok(Either::Right(stream_fallback_chat(
                fallback,
                &gen_inputs,
                active_model_id,
            )));
        }
        let response = fallback_chat_response(active_model_id, &fallback);
        return Ok(Either::Left(Json(response)));
    }
    debug!(prompt = %prompt, "Prepared chat prompt");
    let max_tokens = req.max_tokens.unwrap_or(state.default_max_new_tokens());
    let mut decode = base_decode_parameters(&gen_inputs, max_tokens);
    apply_decode_overrides(
        &mut decode,
        req.do_sample,
        req.temperature,
        req.top_p,
        req.top_k,
        req.repetition_penalty,
        req.no_repeat_ngram_size,
        req.seed,
        req.use_cache,
    );
    if req.stream.unwrap_or(false) {
        let stream_inputs = gen_inputs.clone();
        let decode_for_task = decode.clone();
        let created = current_timestamp();
        let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
        let (sender, rx) = mpsc::unbounded_channel();
        let stream = into_event_stream(rx);
        let context = StreamContext {
            sender,
            kind: StreamKind::Chat {
                completion_id: completion_id.clone(),
                model: active_model_id.clone(),
                created,
            },
        };
        let task_context = context.clone();
        rocket::tokio::spawn(async move {
            let _ = generate_async(
                stream_inputs,
                prompt,
                images,
                decode_for_task,
                Some(task_context),
            )
            .await;
        });
        return Ok(Either::Right(stream));
    }
    let generation = generate_async(gen_inputs, prompt, images, decode, None).await?;
    let created = current_timestamp();
    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".into(),
        created,
        model: active_model_id,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageResponse {
                role: "assistant".into(),
                content: generation.text.clone(),
            },
            finish_reason: "stop".into(),
        }],
        usage: Usage {
            prompt_tokens: generation.prompt_tokens,
            completion_tokens: generation.response_tokens,
            total_tokens: generation.prompt_tokens + generation.response_tokens,
        },
    };
    Ok(Either::Left(Json(response)))
}

pub fn v1_routes() -> Vec<Route> {
    routes![
        health,
        options_models,
        list_models,
        responses_endpoint,
        chat_completions_endpoint
    ]
}

fn apply_decode_overrides(
    params: &mut DecodeParameters,
    do_sample: Option<bool>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    no_repeat_ngram_size: Option<usize>,
    seed: Option<u64>,
    use_cache: Option<bool>,
) {
    if let Some(sample) = do_sample {
        params.do_sample = sample;
    }
    if let Some(temp) = temperature {
        params.temperature = temp;
    }
    if let Some(prob) = top_p {
        params.top_p = if prob < 1.0 { Some(prob) } else { None };
    }
    if let Some(k) = top_k {
        params.top_k = if k == 0 { None } else { Some(k) };
    }
    if let Some(penalty) = repetition_penalty {
        params.repetition_penalty = penalty;
    }
    if let Some(size) = no_repeat_ngram_size {
        params.no_repeat_ngram_size = if size == 0 { None } else { Some(size) };
    }
    if let Some(seed) = seed {
        params.seed = Some(seed);
    }
    if let Some(use_cache) = use_cache {
        params.use_cache = use_cache;
    }
}
fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|dur| dur.as_secs() as i64)
        .unwrap_or_default()
}

fn prompt_missing_image(prompt: &str) -> bool {
    !prompt.contains("<image>")
}

fn missing_image_markdown() -> String {
    "⚠️ **Image Required**\n\n- This OCR backend expects at least one `<image>` placeholder or attached image.\n- Please include `input_image` / `image_url`, or add `<image>` inside the prompt.\n\n---\n\n⚠️ **需要图像输入**\n\n- 当前 OCR 模型需要至少一个 `<image>` 占位符或实际图片。\n- 请在请求中附带 `input_image`/`image_url`，或在 prompt 中插入 `<image>`。".into()
}

fn fallback_response_response(model: String, text: &str) -> ResponsesResponse {
    let created = current_timestamp();
    ResponsesResponse {
        id: format!("resp-{}", Uuid::new_v4()),
        object: "response".into(),
        created,
        model,
        output: vec![ResponseOutput {
            id: format!("msg-{}", Uuid::new_v4()),
            r#type: "message".into(),
            role: "assistant".into(),
            content: vec![ResponseContent {
                r#type: "output_text".into(),
                text: text.to_string(),
            }],
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }
}

fn fallback_chat_response(model: String, text: &str) -> ChatCompletionResponse {
    let created = current_timestamp();
    ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".into(),
        created,
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageResponse {
                role: "assistant".into(),
                content: text.to_string(),
            },
            finish_reason: "stop".into(),
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }
}

fn stream_fallback_response(
    text: String,
    inputs: &GenerationInputs,
    model: String,
) -> BoxEventStream {
    let response_id = format!("resp-{}", Uuid::new_v4());
    let output_id = format!("msg-{}", Uuid::new_v4());
    let created = current_timestamp();
    let (sender, rx) = mpsc::unbounded_channel();
    let stream = into_event_stream(rx);
    let context = StreamContext {
        sender,
        kind: StreamKind::Responses {
            response_id,
            output_id,
            model,
            created,
        },
    };
    let controller = StreamController::new(Arc::clone(&inputs.tokenizer), context);
    controller.send_initial();
    controller.emit_fallback(&text);
    stream
}

fn stream_fallback_chat(text: String, inputs: &GenerationInputs, model: String) -> BoxEventStream {
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = current_timestamp();
    let (sender, rx) = mpsc::unbounded_channel();
    let stream = into_event_stream(rx);
    let context = StreamContext {
        sender,
        kind: StreamKind::Chat {
            completion_id,
            model,
            created,
        },
    };
    let controller = StreamController::new(Arc::clone(&inputs.tokenizer), context);
    controller.send_initial();
    controller.emit_fallback(&text);
    stream
}
