- Removed unused serde::Serialize import from crates/pipeline/src/observer.rs to clear warning.
- In tests, replaced `tempfile` with a local std-only temp-dir helper (`std::env::temp_dir` + pid + timestamp + atomic counter) and best-effort cleanup in `Drop` to avoid dependency requirements.
- `OcrRuntimeBuilder::build()` now resolves layers first, then applies builder-level `device`/`precision` as highest-priority runtime overrides before constructing `OcrModelManager`.
- `OcrModelManager::load` now implements std-only true singleflight with `Mutex<HashMap<...>> + Condvar`: only the winner performs real model loading, waiters block on the same entry and reuse the published result.
- Pipeline manager cache stores successful `OcrPipelineHandle` only; failed loads do not poison cache and can be retried in future calls while still being deduplicated within the same in-flight wave.
- Pipeline generate tests can stay fully offline by constructing a minimal `tokenizers::Tokenizer` from inline JSON via `Tokenizer::from_bytes` (WordLevel + `[UNK]`), avoiding tokenizer files/downloads and extra deps.
- `OcrPipelineHandle` should expose `generate` delegation and `AsRef<OcrPipeline>` so server-side shared handles can call generation directly without reaching into internals.
- Trait guarantees for shared handles are best validated in integration tests with compile-time helpers (`assert_send_sync`, `assert_clone`) plus a lightweight dummy-engine runtime call.
- Stage 2E CLI Cargo migration: keep CLI feature names stable while forwarding each to `deepseek-ocr-pipeline/<same-feature>` and remove direct CLI deps on core/config/assets/infer crates to enforce the entrypoint dependency boundary.
- CLI migration to pipeline runtime can preserve existing token streaming/quiet behavior by cloning the pipeline tokenizer (`handle.pipeline().tokenizer()?`) and computing stdout deltas from progressively decoded token prefixes while calling `OcrPipelineHandle::generate`.
- For `bench-metrics` compatibility during CLI decoupling, pipeline should re-export `deepseek_ocr_core` so CLI benchmark recorder wiring can route through `deepseek_ocr_pipeline::deepseek_ocr_core::benchmark::*` without direct core dependency.

- Task 14 (server manifest migration): server Cargo.toml now routes internal runtime feature flags (metal/accelerate/flash-attn/cuda/mkl) through deepseek-ocr-pipeline and removes direct core/config/assets/infer dependencies; cargo check currently fails until server source imports are migrated to pipeline re-exports (follow-up tasks 15/16).

- Server `state.rs` migration can preserve the one-model cache contract while switching load path to `OcrRuntimeBuilder` + `OcrModelManager::load` by storing only `OcrPipelineHandle` as source-of-truth and deriving request tokenizer/settings from that handle + resolved config layers.
- Pipeline now re-exports `deepseek_ocr_config` so server can import config types via `deepseek_ocr_pipeline::deepseek_ocr_config::*` and keep the dependency boundary centered on the pipeline facade.

- Server source imports should consume / via  re-exports to preserve the pipeline-only dependency boundary.

- Server source imports should consume deepseek_ocr_core/deepseek_ocr_config via deepseek_ocr_pipeline re-exports to preserve the pipeline-only dependency boundary.
- Server `generation.rs` can migrate to pipeline API without route/schema changes by keeping `OcrPromptMessage` at boundaries, converting it to `Vec<deepseek_ocr_pipeline::OcrMessage>` right before `OcrRequest`, and calling `OcrPipelineHandle::generate` with the existing stream callback plumbing.
- Hermetic server API contract tests can validate OpenAI-shape compatibility without loading any model by mounting Rocket-local contract routes that reuse production request parsing (`collect_prompt_inputs`) and production fallback payload builders (`fallback_*_response`) to exercise the same no-image behavior.
- Task 12 debug/bench compatibility: kept CLI debug JSON schema untouched (`rendered_prompt`, `prompt_tokens`, `generated_len`, `tokens`, `decoded`, `normalized`) and updated bench-only disabled-feature error text in `bench.rs` to reference `deepseek-ocr-cli --features bench-metrics` while preserving benchmark event capture/summary formatting and pipeline re-export usage (`deepseek_ocr_pipeline::deepseek_ocr_core::benchmark::*`).
- Task 13 CLI behavior contract tests can stay hermetic by executing the compiled binary with `std::process::Command` and focusing on fast failure-path assertions (missing prompt, bench feature gate, `<image>` slot mismatch), while gating model-dependent checks behind a local-weights existence probe and cfg-gating debug JSON schema validation on `cli-debug`.
- Pushed branch `feature/refactor` with commit `chore: Stage 2E 完成` and created/pushed tag `v0.6.0-stage2e`.
