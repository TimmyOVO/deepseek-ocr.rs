# Issues

- After removing direct core/config/infer deps from `crates/cli/Cargo.toml`, `cargo check -p deepseek-ocr-cli` currently fails because `crates/cli/src/*.rs` still imports `deepseek_ocr_core`, `deepseek_ocr_config`, and `deepseek_ocr_infer_*` directly; source migration to pipeline facade is required in the next task.
- LSP diagnostics cannot run on `Cargo.toml` in this environment because no TOML LSP server is configured.
- `state.rs` now compiles against pipeline-only imports, but full `cargo check -p deepseek-ocr-server` still fails in other server modules (`app.rs`, `args.rs`, `generation.rs`, `models.rs`, `routes.rs`, `stream.rs`) that still import removed direct `deepseek_ocr_config`/`deepseek_ocr_core` paths.
- After migrating `crates/server/src/generation.rs`, `cargo check -p deepseek-ocr-server` still fails in `crates/server/src/app.rs` due to pre-existing `AppState::bootstrap` call-site mismatch (6 args passed to 2-arg signature); this blocker is outside Task 16 scope.
- Cargo cannot condition target dependencies on feature flags ( in target tables is ignored), so manifest-only gating was insufficient for flash-attn imports and required matching Rust cfg target gating.
- Cargo target dependency cfg cannot be keyed by crate feature flags in a working way, so manifest-only gating was insufficient for flash-attn and required matching Rust cfg target gating.
