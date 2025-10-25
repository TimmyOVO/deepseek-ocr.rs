# deepseek-ocr.rs 🚀

Rust implementation of the DeepSeek-OCR inference stack with a fast CLI and an OpenAI-compatible HTTP server. The workspace packages the vision-language model, prompt tooling, and serving layer so you can build document understanding pipelines that run locally on CPU or Apple Metal.

> 在找中文文档? 看这里 [中文](README_CN.md).

## Inside `crates/core` 🔬
- **Vision preprocessing** – `prepare_vision_input_from_image` builds a square global canvas with letterboxing (`build_global_view`) and, when crop mode is enabled, applies `dynamic_preprocess` tiling to produce high-resolution local crops plus optional thumbnails.
- **SAM + CLIP fusion** – each view is normalised via `image_to_tensor`, pushed through the Candle ports of SAM (`SamBackbone`) and CLIP-L (`ClipVisionModel`), then flattened with `build_clip_sam_tokens` so the features stay spatially aligned.
- **Projector & layout tokens** – the custom `ImageProjector` linearly maps concatenated SAM/CLIP channels into the language hidden size while injecting learned `image_newline`/`view_separator` tokens to preserve grid structure, yielding the multimodal embeddings used during decoding.
- **Tokenizer alignment** – `build_prompt_tokens` synthesises `<image>` spans whose length exactly matches the projected token count (global + local grids), ensuring OpenAI-style prompts remain consistent even after chat history pruning.
- **Decoder & caching** – the text stack is a Candle reimplementation of DeepSeek-V2 (`DeepseekLanguageModel`) with optional FlashAttention, rotary position embeddings, and `DynamicCache` guards so both the CLI and server can stream tokens efficiently.
- **Observability & parity** – debug builds expose CLIP/SAM traces (`VisionDebugFeatures`) so we can diff intermediate tensors against the PyTorch reference; most stages are already numerically aligned, and the few remaining deltas (mainly projector normalisation + vision tiling) are tracked on the roadmap for upcoming releases.

## Why Rust? 💡
The original DeepSeek-OCR ships as a Python + Transformers stack—powerful, but hefty to deploy and awkward to embed. Rewriting the pipeline in Rust gives us:
- Smaller deployable artifacts with zero Python runtime or conda baggage.
- Memory-safe, thread-friendly infrastructure that blends into native Rust backends.
- Unified tooling (CLI + server) running on Candle + Rocket without the Python GIL overhead.
- Drop-in compatibility with OpenAI-style clients while tuned for single-turn OCR prompts.

## Technical Stack ⚙️
- **Candle** for tensor compute, with Metal and CUDA backends and FlashAttention support.
- **Rocket** + async streaming for OpenAI-compatible `/v1/responses` and `/v1/chat/completions`.
- **tokenizers** (Hugging Face) wrapped by `crates/assets` for deterministic caching.
- **Pure Rust vision/prompt pipeline** shared by CLI and server to avoid duplicated logic.

## Advantages over the Python Release 🥷
- Faster cold-start on Apple Silicon, lower RSS, and native binary distribution.
- Deterministic Hugging Face asset download + verification built into the workspace.
- Automatic single-turn chat compaction so OCR outputs stay stable even when clients send history.
- Ready-to-use OpenAI compatibility for tools like Open WebUI without adapters.

## Highlights ✨
- **One repo, two entrypoints** – a batteries-included CLI for batch jobs and a Rocket-based server that speaks `/v1/responses` and `/v1/chat/completions`.
- **Works out of the box** – pulls model weights, configs, and tokenizer from Hugging Face on first run.
- **Optimised for Apple Silicon** – optional Metal backend with FP16 execution for real-time OCR on laptops.
- **OpenAI client compatibility** – drop-in replacement for popular SDKs; the server automatically collapses chat history to the latest user turn for OCR-friendly prompts.

## Quick Start 🏁

### Prerequisites
- Rust 1.78+ (edition 2024 support)
- Git
- Optional: Apple Silicon running macOS 13+ for Metal acceleration
- (Recommended) Hugging Face account with `HF_TOKEN` when pulling from the `deepseek-ai/DeepSeek-OCR` repo

### Clone the Workspace
```bash
git clone https://github.com/TimmyOVO/deepseek-ocr.rs.git
cd deepseek-ocr.rs
cargo fetch
```

### Model Assets
The first invocation of the CLI or server downloads the config, tokenizer, and `model-00001-of-000001.safetensors` (~6.3GB) into `DeepSeek-OCR/`. To prefetch manually:
```bash
cargo run -p deepseek-ocr-cli -- --help # triggers asset download
```
Set `HF_HOME` or `HF_TOKEN` if you store Hugging Face caches elsewhere. The full model package is ~6.3GB on disk and typically requires ~13GB of RAM headroom during inference (model + activations).

## Command-Line Interface 🖥️
Build and run directly from the workspace:
```bash
cargo run -p deepseek-ocr-cli -- \
  --prompt "<image>\n<|grounding|>Convert this receipt to markdown." \
  --image baselines/sample/images/test.png \
  --device cpu --max-new-tokens 512
```

Install the CLI as a binary:
```bash
cargo install --path crates/cli
deepseek-ocr-cli --help
```

Key flags:
- `--prompt` / `--prompt-file`: text with `<image>` slots
- `--image`: path(s) matching `<image>` placeholders
- `--device` and `--dtype`: choose `metal` + `f16` on Apple Silicon
- `--max-new-tokens`: decoding budget

## HTTP Server ☁️
Launch an OpenAI-compatible endpoint:
```bash
cargo run -p deepseek-ocr-server -- \
  --host 0.0.0.0 --port 8000 \
  --device cpu --max-new-tokens 512
```

Notes:
- Use `data:` URLs or remote `http(s)` links; local paths are rejected.
- The server collapses multi-turn chat inputs to the latest user message to keep prompts OCR-friendly.
- Works out of the box with tools such as [Open WebUI](https://github.com/open-webui/open-webui) or any OpenAI-compatible client—just point the base URL to your server (`http://localhost:8000/v1`) and select the `deepseek-ocr` model.
- Adjust the request body limit with Rocket config if you routinely send large images.

![Open WebUI connected to deepseek-ocr.rs](./assets/sample_1.png)

## Metal Acceleration ⚡
- Available on macOS 13+ with Apple Silicon.
- Pass `--device metal --dtype f16` to either CLI or server.
- For best throughput, build the release profile: `cargo build --release -p deepseek-ocr-cli`.
- Combine with `--max-new-tokens` and crop options to tune latency.

## Repository Layout 🗂️
- `crates/core` – shared inference pipeline, model loaders, conversation templates.
- `crates/cli` – command-line frontend (`deepseek-ocr-cli`).
- `crates/server` – Rocket server exposing OpenAI-compatible endpoints.
- `crates/assets` – asset management (configuration, tokenizer, Hugging Face download helpers).
- `baselines/` – reference inputs and outputs for regression testing.

## Troubleshooting 🛠️
- **Weights download fails** – export `HF_TOKEN=<your-token>` and retry. Assets land in `~/.cache/huggingface` by default.
- **Slow first response** – model load and Metal warm-up happen on the initial request; later runs are faster.
- **Large image rejection** – increase Rocket JSON limits in `crates/server/src/main.rs` or downscale the input.

## Roadmap 🗺️
- ✅ Apple Metal backend with FP16 support and CLI/server parity on macOS.
- 🔄 **Parity polish** – finish projector normalisation + crop tiling alignment; extend intermediate-tensor diff suite beyond the current sample baseline.
- 🔄 **Grounding & streaming** – port the Python post-processing helpers (box extraction, markdown polish) and refine SSE streaming ergonomics.
- 🔄 **Cross-platform acceleration** – stabilise the Windows CUDA/FlashAttention prototype, bring up Vulkan/Metal auto-detection, and add opt-in GPU benchmarks.
- 🔄 **Packaging & Ops** – ship binary releases with deterministic asset checksums, richer logging/metrics, and Helm/docker references for server deploys.
- 🔜 **Structured outputs** – optional JSON schema tools for downstream automation once parity gaps close.

## License 📄
This repository inherits the licenses of its dependencies and the upstream DeepSeek-OCR model. Refer to `DeepSeek-OCR/LICENSE` for model terms and apply the same restrictions to downstream use.
