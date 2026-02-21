use std::fmt;
use std::time::Duration;

use crate::{ModelKind, OcrModelId};

/// Pipeline-level events intended for observability, profiling, and debugging.
///
/// 设计目标：
/// - CLI/Server/第三方应用都可以通过一个统一 observer 接口获取关键信息；
/// - 事件语义保持稳定，内部实现可变；
/// - 不把底层 `infer-*` 的细节直接暴露到应用层。
#[derive(Debug, Clone)]
pub enum OcrPipelineEvent {
    ConfigLoaded {
        config_path: Option<String>,
        active_model: OcrModelId,
    },

    ResourcesPrepared {
        model_id: OcrModelId,
        config: String,
        tokenizer: String,
        weights: String,
        snapshot: Option<String>,
    },

    ModelLoadStarted {
        model_id: OcrModelId,
    },

    ModelLoadFinished {
        model_id: OcrModelId,
        kind: ModelKind,
        flash_attention: bool,
        duration: Duration,
    },

    GenerationStarted {
        model_id: OcrModelId,
        max_new_tokens: usize,
    },

    GenerationFinished {
        model_id: OcrModelId,
        prompt_tokens: usize,
        response_tokens: usize,
        duration: Duration,
    },
}

fn format_duration_s(duration: &Duration) -> String {
    format!("{:.3}s", duration.as_secs_f64())
}

impl serde::Serialize for OcrPipelineEvent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use OcrPipelineEvent::*;

        match self {
            ConfigLoaded {
                config_path,
                active_model,
            } => {
                #[derive(serde::Serialize)]
                struct Event<'a> {
                    kind: &'static str,
                    config_path: &'a Option<String>,
                    active_model: &'a OcrModelId,
                }

                Event {
                    kind: "config_loaded",
                    config_path,
                    active_model,
                }
                .serialize(serializer)
            }
            ResourcesPrepared {
                model_id,
                config,
                tokenizer,
                weights,
                snapshot,
            } => {
                #[derive(serde::Serialize)]
                struct Event<'a> {
                    kind: &'static str,
                    model_id: &'a OcrModelId,
                    config: &'a str,
                    tokenizer: &'a str,
                    weights: &'a str,
                    snapshot: &'a Option<String>,
                }

                Event {
                    kind: "resources_prepared",
                    model_id,
                    config,
                    tokenizer,
                    weights,
                    snapshot,
                }
                .serialize(serializer)
            }
            ModelLoadStarted { model_id } => {
                #[derive(serde::Serialize)]
                struct Event<'a> {
                    kind: &'static str,
                    model_id: &'a OcrModelId,
                }

                Event {
                    kind: "model_load_started",
                    model_id,
                }
                .serialize(serializer)
            }
            ModelLoadFinished {
                model_id,
                kind,
                flash_attention,
                duration,
            } => {
                #[derive(serde::Serialize)]
                struct Event<'a> {
                    kind: &'static str,
                    model_id: &'a OcrModelId,
                    model_kind: ModelKind,
                    flash_attention: bool,
                    duration_s: String,
                }

                Event {
                    kind: "model_load_finished",
                    model_id,
                    model_kind: *kind,
                    flash_attention: *flash_attention,
                    duration_s: format_duration_s(duration),
                }
                .serialize(serializer)
            }
            GenerationStarted {
                model_id,
                max_new_tokens,
            } => {
                #[derive(serde::Serialize)]
                struct Event<'a> {
                    kind: &'static str,
                    model_id: &'a OcrModelId,
                    max_new_tokens: usize,
                }

                Event {
                    kind: "generation_started",
                    model_id,
                    max_new_tokens: *max_new_tokens,
                }
                .serialize(serializer)
            }
            GenerationFinished {
                model_id,
                prompt_tokens,
                response_tokens,
                duration,
            } => {
                #[derive(serde::Serialize)]
                struct Event<'a> {
                    kind: &'static str,
                    model_id: &'a OcrModelId,
                    prompt_tokens: usize,
                    response_tokens: usize,
                    duration_s: String,
                }

                Event {
                    kind: "generation_finished",
                    model_id,
                    prompt_tokens: *prompt_tokens,
                    response_tokens: *response_tokens,
                    duration_s: format_duration_s(duration),
                }
                .serialize(serializer)
            }
        }
    }
}

impl fmt::Display for OcrPipelineEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use OcrPipelineEvent::*;

        match self {
            ConfigLoaded {
                config_path,
                active_model,
            } => {
                if let Some(path) = config_path {
                    write!(f, "ConfigLoaded {active_model} from {path}")
                } else {
                    write!(f, "ConfigLoaded {active_model}")
                }
            }
            ResourcesPrepared {
                model_id,
                config,
                tokenizer,
                weights,
                snapshot,
            } => {
                match snapshot {
                    Some(s) => write!(
                        f,
                        "ResourcesPrepared {model_id} config={config} tokenizer={tokenizer} weights={weights} snapshot={s}"
                    ),
                    None => write!(
                        f,
                        "ResourcesPrepared {model_id} config={config} tokenizer={tokenizer} weights={weights}"
                    ),
                }
            }
            ModelLoadStarted { model_id } => write!(f, "ModelLoadStarted {model_id}"),
            ModelLoadFinished {
                model_id,
                kind,
                flash_attention,
                duration,
            } => write!(
                f,
                "ModelLoadFinished {model_id} kind={kind:?} flash_attention={} in {}",
                flash_attention,
                format_duration_s(duration)
            ),
            GenerationStarted {
                model_id,
                max_new_tokens,
            } => write!(f, "GenerationStarted {model_id} max_new_tokens={max_new_tokens}"),
            GenerationFinished {
                model_id,
                prompt_tokens,
                response_tokens,
                duration,
            } => write!(
                f,
                "GenerationFinished {model_id} prompt_tokens={prompt_tokens} response_tokens={response_tokens} in {}",
                format_duration_s(duration)
            ),
        }
    }
}

/// Observer interface for pipeline lifecycle and generation metrics.
///
/// 默认实现应是 no-op，避免上层必须关心观察逻辑。
pub trait OcrPipelineObserver: Send + Sync {
    fn on_event(&self, _event: &OcrPipelineEvent) {}
}

#[derive(Debug, Default)]
pub struct NoopObserver;

impl OcrPipelineObserver for NoopObserver {}
