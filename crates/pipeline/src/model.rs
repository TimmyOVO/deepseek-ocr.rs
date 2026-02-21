use std::fmt;

use anyhow::{ensure, Result};

/// Strongly-typed model identifier.
///
/// 设计目标：
/// - 配置文件中依然使用单一 `model_id` 字符串（对用户友好）；
/// - 进入 Rust 代码后，尽早把 `String` 收敛为强类型，避免到处传裸字符串；
/// - 提供最基础的约束（非空、无空白符），把错误前置。
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
pub struct OcrModelId(String);

impl OcrModelId {
    /// Parse a model id from configuration / CLI input.
    pub fn from_model_id(model_id: String) -> Result<Self> {
        let trimmed = Self::validate(&model_id)?;
        Ok(Self(trimmed.to_string()))
    }

    fn validate(model_id: &str) -> Result<&str> {
        let trimmed = model_id.trim();
        ensure!(!trimmed.is_empty(), "model id must be non-empty");
        ensure!(
            !trimmed.chars().any(char::is_whitespace),
            "model id must not contain whitespace"
        );
        Ok(trimmed)
    }

    pub fn known_models() -> &'static [&'static str] {
        &[
            "deepseek-ocr",
            "deepseek-ocr-q4k",
            "deepseek-ocr-q6k",
            "deepseek-ocr-q8k",
            "paddleocr-vl",
            "paddleocr-vl-q4k",
            "paddleocr-vl-q6k",
            "paddleocr-vl-q8k",
            "dots-ocr",
            "dots-ocr-q4k",
            "dots-ocr-q6k",
            "dots-ocr-q8k",
            "glm-ocr",
        ]
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for OcrModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl TryFrom<String> for OcrModelId {
    type Error = anyhow::Error;

    fn try_from(value: String) -> Result<Self> {
        Self::from_model_id(value)
    }
}

impl TryFrom<&str> for OcrModelId {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self> {
        Self::from_model_id(value.to_string())
    }
}
