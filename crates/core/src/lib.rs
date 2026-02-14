pub mod attention;
pub mod benchmark;
pub mod cache;
pub mod config;
pub mod conversation;
pub mod inference;
pub mod ocr_inference_engine;
pub mod prompt_tokens;
pub mod runtime;
pub mod runtime_backend;
pub mod sampling;
pub mod streaming;
pub mod tensor;

pub use inference::{
    DecodeOutcome, DecodeParameters, DecodeParametersPatch, ModelKind, ModelLoadArgs, OcrEngine,
    VisionSettings, normalize_text, render_conversation, render_prompt,
};
pub use ocr_inference_engine::{
    DefaultModelSemantics, ModelSemantics, OcrInferenceEngine, OcrInferenceRequest,
    OcrInferenceResult, OcrPromptInput, OcrPromptMessage, OcrPromptRenderRequest,
    OcrPromptRole, PreparedInputs, RenderedPrompt,
};
pub use prompt_tokens::{
    IMAGE_PLACEHOLDER, PromptBuildOptions, PromptTokenSequence, build_prompt_tokens_with,
    grid_token_count,
};
pub use runtime_backend::{
    BackendCaps, BackendHooks, KvReadView, KvStateBackend, PromptScopeGuard, RopeState,
    RuntimeBackend, RuntimeBackendKind, RuntimeEngine, RuntimeModelSpec, RuntimeState,
    RuntimeStateGuard, RuntimeStepOutput, RuntimeStepRequest, SamplingOutput, SamplingRequest,
    WorkspaceArena,
};

// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;

#[cfg(feature = "memlog")]
pub mod memlog;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Placeholder entry point while components are being ported from Python.
pub fn init() {
    // Initialization logic (e.g., logger setup) will live here.
}
