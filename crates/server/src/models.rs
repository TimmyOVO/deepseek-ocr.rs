use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub output: Vec<ResponseOutput>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ResponseOutput {
    pub id: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub role: String,
    pub content: Vec<ResponseContent>,
}

#[derive(Debug, Serialize)]
pub struct ResponseContent {
    #[serde(rename = "type")]
    pub r#type: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessageResponse,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageResponse {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    #[serde(default)]
    pub input: Vec<ApiMessage>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub no_repeat_ngram_size: Option<usize>,
    #[serde(default)]
    pub do_sample: Option<bool>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub use_cache: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ApiMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub no_repeat_ngram_size: Option<usize>,
    #[serde(default)]
    pub do_sample: Option<bool>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub use_cache: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ApiMessage {
    pub role: String,
    #[serde(default)]
    pub content: MessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<MessagePart>),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagePart {
    Text { text: String },
    InputText { text: String },
    ImageUrl { image_url: ImagePayload },
    InputImage { image_url: ImagePayload },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ImagePayload {
    Simple(String),
    Detailed { url: String },
}

impl ImagePayload {
    pub fn url(&self) -> &str {
        match self {
            ImagePayload::Simple(u) => u,
            ImagePayload::Detailed { url } => url,
        }
    }
}
