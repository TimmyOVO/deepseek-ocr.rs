#[macro_use]
extern crate rocket;

use rocket::{
    Build, Rocket,
    http::{ContentType, Status},
    local::asynchronous::Client,
    serde::json::Json,
};
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};

mod error {
    pub use deepseek_ocr_server::error::*;
}

mod generation {
    pub use deepseek_ocr_server::generation::*;
}

mod models {
    pub use deepseek_ocr_server::models::*;
}

fn unix_timestamp_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

fn missing_image_markdown() -> String {
    [
        "⚠️ **Image Required**",
        "",
        "OCR extraction needs an image input.",
        "Please send at least one `image_url` in the message content.",
    ]
    .join("\n")
}

fn fallback_response_response(model: String, text: &str) -> models::ResponsesResponse {
    models::ResponsesResponse {
        id: format!("resp_contract_{}", uuid::Uuid::new_v4()),
        object: "response".to_string(),
        created: unix_timestamp_now(),
        model,
        output: vec![models::ResponseOutput {
            id: format!("out_contract_{}", uuid::Uuid::new_v4()),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![models::ResponseContent {
                r#type: "output_text".to_string(),
                text: text.to_string(),
            }],
        }],
        usage: models::Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }
}

fn fallback_chat_response(model: String, text: &str) -> models::ChatCompletionResponse {
    models::ChatCompletionResponse {
        id: format!("chatcmpl_contract_{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: unix_timestamp_now(),
        model,
        choices: vec![models::ChatChoice {
            index: 0,
            message: models::ChatMessageResponse {
                role: "assistant".to_string(),
                content: text.to_string(),
            },
            finish_reason: "stop".to_string(),
        }],
        usage: models::Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }
}

#[get("/models")]
fn models_contract() -> Json<models::ModelsResponse> {
    Json(models::ModelsResponse {
        object: "list".into(),
        data: vec![models::ModelInfo {
            id: "deepseek-ocr".into(),
            object: "model".into(),
            created: 0,
            owned_by: "deepseek-ocr".into(),
        }],
    })
}

#[post("/responses", format = "json", data = "<req>")]
async fn responses_contract(
    req: Json<models::ResponsesRequest>,
) -> Result<Json<models::ResponsesResponse>, error::ApiError> {
    let parsed = generation::collect_prompt_inputs(&req.input)?;
    if parsed.images.is_empty() {
        let fallback = missing_image_markdown();
        return Ok(Json(fallback_response_response(
            req.model.clone(),
            &fallback,
        )));
    }
    Err(error::ApiError::BadRequest(
        "test contract route expects text-only prompt for hermetic fallback path".into(),
    ))
}

#[post("/chat/completions", format = "json", data = "<req>")]
async fn chat_completions_contract(
    req: Json<models::ChatCompletionRequest>,
) -> Result<Json<models::ChatCompletionResponse>, error::ApiError> {
    let parsed = generation::collect_prompt_inputs(&req.messages)?;
    if parsed.images.is_empty() {
        let fallback = missing_image_markdown();
        return Ok(Json(fallback_chat_response(
            req.model.clone(),
            &fallback,
        )));
    }
    Err(error::ApiError::BadRequest(
        "test contract route expects text-only prompt for hermetic fallback path".into(),
    ))
}

fn contract_rocket() -> Rocket<Build> {
    rocket::build().mount(
        "/v1",
        routes![models_contract, responses_contract, chat_completions_contract],
    )
}

async fn read_json(response: rocket::local::asynchronous::LocalResponse<'_>) -> Value {
    response
        .into_string()
        .await
        .map(|s| serde_json::from_str::<Value>(&s).expect("response must be valid JSON"))
        .expect("response body should exist")
}

#[rocket::async_test]
async fn models_returns_expected_shape() {
    let client = Client::tracked(contract_rocket())
        .await
        .expect("client should build");

    let response = client.get("/v1/models").dispatch().await;
    assert_eq!(response.status(), Status::Ok);

    let body = read_json(response).await;
    assert_eq!(body["object"], "list");
    assert!(body["data"].is_array());

    let first = &body["data"][0];
    assert!(first["id"].is_string());
    assert_eq!(first["object"], "model");
    assert!(first["created"].is_i64() || first["created"].is_u64());
    assert!(first["owned_by"].is_string());
}

#[rocket::async_test]
async fn responses_non_stream_returns_fallback_shape_without_model_generation() {
    let client = Client::tracked(contract_rocket())
        .await
        .expect("client should build");

    let payload = json!({
        "model": "deepseek-ocr",
        "input": [{"role": "user", "content": "extract text"}],
        "stream": false
    });

    let response = client
        .post("/v1/responses")
        .header(ContentType::JSON)
        .body(payload.to_string())
        .dispatch()
        .await;
    assert_eq!(response.status(), Status::Ok);

    let body = read_json(response).await;
    assert_eq!(body["object"], "response");
    assert_eq!(body["output"][0]["role"], "assistant");
    assert!(body["output"][0]["content"][0]["text"]
        .as_str()
        .is_some_and(|text| text.contains("Image Required")));
}

#[rocket::async_test]
async fn chat_completions_non_stream_returns_fallback_shape_without_model_generation() {
    let client = Client::tracked(contract_rocket())
        .await
        .expect("client should build");

    let payload = json!({
        "model": "deepseek-ocr",
        "messages": [{"role": "user", "content": "extract text"}],
        "stream": false
    });

    let response = client
        .post("/v1/chat/completions")
        .header(ContentType::JSON)
        .body(payload.to_string())
        .dispatch()
        .await;
    assert_eq!(response.status(), Status::Ok);

    let body = read_json(response).await;
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(body["choices"][0]["message"]["content"]
        .as_str()
        .is_some_and(|text| text.contains("Image Required")));
}

#[rocket::async_test]
async fn bad_request_returns_error_message_and_type_shape() {
    let client = Client::tracked(contract_rocket())
        .await
        .expect("client should build");

    let payload = json!({
        "model": "deepseek-ocr",
        "input": []
    });

    let response = client
        .post("/v1/responses")
        .header(ContentType::JSON)
        .body(payload.to_string())
        .dispatch()
        .await;
    assert_eq!(response.status(), Status::BadRequest);

    let body = read_json(response).await;
    assert!(body["error"]["message"].is_string());
    assert!(body["error"]["type"].is_string());
}
