# Decisions

- Keep `cargo check --workspace --all-targets --all-features` green on macOS by making CUDA/MKL/flash-attn feature activation target-aware at the crate manifest layer, so workspace-wide feature sweeps remain a reliable CI/local gate regardless of host GPU toolchains.
- Treat `cuda`, `flash-attn`, and `mkl` as no-op feature wires on macOS by binding their optional dependencies only under `cfg(any(target_os = "linux", target_os = "windows"))`; supported targets still resolve and build the real backend crates when those features are enabled.
- Cargo feature names remain unchanged (, ), but CUDA-triggering optional deps are target-gated to Linux/Windows in manifests so macOS treats these feature toggles as no-op dependency-wise.
- To keep  green on macOS, flash-attn Rust cfg gates were tightened to  so feature-enabled code only compiles where the optional dependency exists.
-  forwarding in affected infer/core manifests was made empty to avoid simultaneous Accelerate+MKL activation under workspace  on macOS (which causes upstream candle duplicate symbol/definition failures).
- Kept feature names unchanged (cuda, flash-attn), while target-gating CUDA-triggering optional dependencies to Linux/Windows so macOS treats those feature toggles as dependency no-ops.
- Kept all-features macOS checks green by tightening flash-attn cfg usage in Rust to require both feature enabled and Linux/Windows target, matching dependency availability.
- Set mkl forwarding to empty in affected infer/core manifests to avoid simultaneous Accelerate plus MKL activation during workspace all-features on macOS, which triggers upstream candle duplicate-definition failures.
