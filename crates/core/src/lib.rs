pub mod benchmark;
pub mod config;
pub mod conversation;
pub mod inference;
pub mod model;
pub mod runtime;
pub mod streaming;
pub mod transformer;
pub mod vision;

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
