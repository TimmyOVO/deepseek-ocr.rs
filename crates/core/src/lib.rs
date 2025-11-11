pub mod benchmark;
pub mod cache;
pub mod conversation;
pub mod inference;
pub mod runtime;
pub mod sampling;
pub mod streaming;
pub mod tensor;

pub use inference::{
    DecodeOutcome, DecodeParameters, ModelKind, ModelLoadArgs, OcrEngine, VisionSettings,
    normalize_text, render_prompt,
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
