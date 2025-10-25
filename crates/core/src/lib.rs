pub mod config;
pub mod conversation;
pub mod inference;
pub mod model;
pub mod runtime;
pub mod transformer;
pub mod vision;

#[cfg(feature = "memlog")]
pub mod memlog;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

pub mod test_utils;

/// Placeholder entry point while components are being ported from Python.
pub fn init() {
    // Initialization logic (e.g., logger setup) will live here.
}
