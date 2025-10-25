use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::Tensor;

/// Tracks live KV cache bytes when the `memlog` feature is enabled.
pub static KV_BYTES: AtomicUsize = AtomicUsize::new(0);
/// Tracks live RoPE table bytes when the `memlog` feature is enabled.
pub static ROPE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Returns the number of bytes occupied by `tensor`.
pub fn tensor_bytes(tensor: &Tensor) -> usize {
    let elems = tensor.shape().dims().iter().copied().product::<usize>();
    elems * tensor.dtype().size_in_bytes()
}

pub fn add_kv(bytes: usize) {
    KV_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

pub fn sub_kv(bytes: usize) {
    KV_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

pub fn set_rope(bytes: usize) {
    ROPE_BYTES.store(bytes, Ordering::Relaxed);
}

/// Emits a simple eprintln! snapshot of current tracked bytes.
pub fn log_snapshot(tag: &str) {
    let kv = KV_BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
    let rope = ROPE_BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
    eprintln!("[memlog] {tag}: kv={kv:.3} MB rope={rope:.3} MB");
}
