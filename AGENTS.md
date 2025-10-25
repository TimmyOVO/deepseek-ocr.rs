# Repository Guidelines

## Project Structure & Module Organization
- `crates/core/src` hosts the shared vision-language pipeline, and `crates/core/tests` keeps the regression suites.
- `crates/cli/src` provides the `deepseek-ocr-cli` entrypoint, while `crates/server/src` exposes the Rocket/OpenAI HTTP surface; both depend on `crates/assets` for tokenizer/config caching.
- Reference assets live in `baselines/` + `assets/`, while downloaded weights land in gitignored `DeepSeek-OCR/`. Keep `metal`/`cuda` feature code inside the owning crate to prevent server bloat.

## Build, Test, and Development Commands
- Format + lint before review:
  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-features -D warnings
  ```
- Core workflows:
  ```bash
  cargo run -p deepseek-ocr-cli -- --prompt "<image>\n<|grounding|>Summarize." --image baselines/sample/images/test.png
  cargo run -p deepseek-ocr-server -- --host 0.0.0.0 --port 8000 --device cpu
  cargo build --release -p deepseek-ocr-cli --features metal   # swap to cuda for NVIDIA GPUs
  ```
  GPU builds: `--features metal` on Apple, `cuda` on NVIDIA. Set `HF_TOKEN=...` if the first run needs private assets.

## Coding Style & Naming Conventions
- Follow `rustfmt` defaults (4-space indent, trailing commas) and keep imports sorted by module depth.
- Apply Rust idioms: snake_case modules/files, PascalCase types (`DeepseekLanguageModel`), SCREAMING_SNAKE consts, and builder-style CLI flag structs. Prefer `anyhow::Result<T>` returns and early `?` error propagation.
- Guard feature-specific code (`#[cfg(feature = "metal")]` / `#[cfg(feature = "cuda")]`) and document unsafe blocks inline with a one-line justification.

## Testing Guidelines
- Run `cargo test --workspace` before every push; CI relies on the same invocation.
- For precision-sensitive work, re-run the baseline-specific suites: `cargo test -p deepseek-ocr-core baseline`. Keep expected tensors/images in `baselines/` and refresh them only when functional parity is proven.
- Name new integration tests after the model component they cover (e.g., `tests/vision_clip.rs`) and include a short doc comment summarizing the scenario.

## Commit & Pull Request Guidelines
- The log mixes sentence-case summaries and Conventional Commits (`feat:…`, `ci:…`); prefer the latter for new work so CI jobs can auto-scope releases.
- Commits should be focused (one feature or fix) and accompanied by updated docs/tests when behavior changes.
- Pull requests must describe: motivation, key commands run (copy the `cargo` lines), expected latency/memory changes, and any asset or baseline updates. Attach CLI/HTTP samples or screenshots when UI-facing behavior shifts.

## Security & Configuration Tips
- Never commit contents of `DeepSeek-OCR/` or personal Hugging Face tokens; rely on environment variables or `.env.local`.
- When testing remote inference, scrub logs before sharing—requests often embed full document text. Prefer redacted snippets when demonstrating issues.
