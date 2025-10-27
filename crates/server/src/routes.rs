use std::time::SystemTime;

use rocket::{Either, Route, State, serde::json::Json, tokio::sync::mpsc};
use tracing::debug;
use uuid::Uuid;

use crate::{
    error::ApiError,
    generation::{convert_messages, generate_async},
    models::{
        ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessageResponse, ModelInfo,
        ModelsResponse, ResponseContent, ResponseOutput, ResponsesRequest, ResponsesResponse,
        Usage,
    },
    state::{AppState, GenerationInputs},
    stream::{BoxEventStream, StreamContext, StreamKind, into_event_stream},
};

#[get("/health")]
pub fn health() -> &'static str {
    "ok"
}

#[get("/models")]
pub fn list_models(state: &State<AppState>) -> Json<ModelsResponse> {
    let now = current_timestamp();
    Json(ModelsResponse {
        object: "list".into(),
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model".into(),
            created: now,
            owned_by: "deepseek".into(),
        }],
    })
}

#[post("/responses", format = "json", data = "<req>")]
pub async fn responses_endpoint(
    state: &State<AppState>,
    req: Json<ResponsesRequest>,
) -> Result<Either<Json<ResponsesResponse>, BoxEventStream>, ApiError> {
    ensure_model(&req.model, &state.model_id)?;
    let gen_inputs = GenerationInputs::from_app(state.inner());
    let (prompt, images) = convert_messages(&req.input)?;
    let max_tokens = req
        .max_output_tokens
        .or(req.max_tokens)
        .unwrap_or(state.max_new_tokens);
    if req.stream.unwrap_or(false) {
        let stream_inputs = gen_inputs.clone();
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
                model: state.model_id.clone(),
                created,
            },
        };
        let task_context = context.clone();
        rocket::tokio::spawn(async move {
            let _ = generate_async(
                stream_inputs,
                prompt,
                images,
                max_tokens,
                Some(task_context),
            )
            .await;
        });
        return Ok(Either::Right(stream));
    }
    let generation = generate_async(gen_inputs, prompt, images, max_tokens, None).await?;
    let created = current_timestamp();
    let response = ResponsesResponse {
        id: format!("resp-{}", Uuid::new_v4()),
        object: "response".into(),
        created,
        model: req.model.clone(),
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
    ensure_model(&req.model, &state.model_id)?;
    let gen_inputs = GenerationInputs::from_app(state.inner());
    let (prompt, images) = convert_messages(&req.messages)?;
    debug!(prompt = %prompt, "Prepared chat prompt");
    let max_tokens = req.max_tokens.unwrap_or(state.max_new_tokens);
    if req.stream.unwrap_or(false) {
        let stream_inputs = gen_inputs.clone();
        let created = current_timestamp();
        let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
        let (sender, rx) = mpsc::unbounded_channel();
        let stream = into_event_stream(rx);
        let context = StreamContext {
            sender,
            kind: StreamKind::Chat {
                completion_id: completion_id.clone(),
                model: state.model_id.clone(),
                created,
            },
        };
        let task_context = context.clone();
        rocket::tokio::spawn(async move {
            let _ = generate_async(
                stream_inputs,
                prompt,
                images,
                max_tokens,
                Some(task_context),
            )
            .await;
        });
        return Ok(Either::Right(stream));
    }
    let generation = generate_async(gen_inputs, prompt, images, max_tokens, None).await?;
    let created = current_timestamp();
    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".into(),
        created,
        model: req.model.clone(),
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
        list_models,
        responses_endpoint,
        chat_completions_endpoint
    ]
}

fn ensure_model(requested: &str, available: &str) -> Result<(), ApiError> {
    if requested == available {
        Ok(())
    } else {
        Err(ApiError::BadRequest(format!(
            "requested model `{requested}` is not available"
        )))
    }
}

fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|dur| dur.as_secs() as i64)
        .unwrap_or_default()
}
