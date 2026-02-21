//! High-level request/response types for OCR inference.

use image::DynamicImage;

use crate::{DecodeParameters, VisionSettings};

/// Callback used to stream decoded token pieces.
///
/// 语义：回调会在生成过程中被调用，参数含义为：
/// - `count`: 当前已生成的 token 总数
/// - `token_ids`: 生成 token id 序列（长度 >= count）
///
/// 说明：该类型保持与现有 core 侧 stream callback 兼容，便于后续迁移实现。
pub type OcrStreamCallback<'a> = Option<&'a dyn Fn(usize, &[i64])>;

/// Role for chat-style prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OcrRole {
    System,
    User,
    Assistant,
}

/// Single message in a chat-style prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrMessage {
    pub role: OcrRole,
    pub content: String,
}

impl OcrMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: OcrRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: OcrRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: OcrRole::Assistant,
            content: content.into(),
        }
    }
}

/// Prompt input accepted by the OCR pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OcrPrompt {
    /// Raw user prompt containing `<image>` placeholders.
    Raw(String),
    /// Already-rendered prompt.
    Rendered(String),
    /// Chat-style messages (Transformers `apply_chat_template`-like).
    Messages(Vec<OcrMessage>),
}

/// High-level OCR request.
///
/// 设计目标：
/// - 作为 pipeline 对外暴露的稳定语义输入；
/// - 未来内部会映射到各模型的 `ModelSemantics` 与统一 runtime 执行环；
/// - 支持 raw prompt 与 messages 两种入口。
#[derive(Debug, Clone)]
pub struct OcrRequest {
    pub prompt: OcrPrompt,

    /// Conversation template name (e.g. `plain`).
    pub template: String,

    /// Optional system prompt.
    pub system_prompt: String,

    /// Images corresponding to `<image>` placeholders.
    pub images: Vec<DynamicImage>,

    pub vision: VisionSettings,
    pub decode: DecodeParameters,
}

/// High-level OCR response.
#[derive(Debug, Clone)]
pub struct OcrResponse {
    pub text: String,
    pub rendered_prompt: String,
    pub prompt_tokens: usize,
    pub response_tokens: usize,
    pub generated_tokens: Vec<i64>,
}
