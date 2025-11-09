# deepseek-ocr.rs 🚀

Rust implementation of the DeepSeek-OCR inference stack with a fast CLI and an OpenAI-compatible HTTP server. The workspace packages multiple OCR backends, prompt tooling, and a serving layer so you can build document understanding pipelines that run locally on CPU, Apple Metal, or (alpha) NVIDIA CUDA GPUs.

> 中文文档请看 [README_CN.md](README_CN.md)。  

> Want ready-made binaries? Latest macOS (Metal-enabled) and Windows bundles live in the [build-binaries workflow artifacts](https://github.com/TimmyOVO/deepseek-ocr.rs/actions/workflows/build-binaries.yml). Grab them from the newest green run.

## Choosing a Model 🔬

| Model | Memory footprint* | Best on | When to pick it |
| --- | --- | --- | --- |
| **DeepSeek‑OCR** | **≈6.3 GB** FP16 weights, **≈13 GB** RAM/VRAM with cache & activations (512-token budget) | Apple Silicon + Metal (FP16), high-VRAM NVIDIA GPUs, 32 GB+ RAM desktops | Highest accuracy, SAM+CLIP global/local context, MoE DeepSeek‑V2 decoder (3 B params, ~570 M active per token). Use when latency is secondary to quality. |
| **PaddleOCR‑VL** | **≈4.7 GB** FP16 weights, **≈9 GB** RAM/VRAM with cache & activations | 16 GB laptops, CPU-only boxes, mid-range GPUs | Dense 0.9 B Ernie decoder with SigLIP vision tower. Faster startup, lower memory, great for batch jobs or lightweight deployments. |

\*Measured from the default FP16 safetensors. Runtime footprint varies with sequence length.

Guidance:

- **Need maximum fidelity, multi-region reasoning, or already have 16–24 GB VRAM?** Use **DeepSeek‑OCR**. The hybrid SAM+CLIP tower plus DeepSeek‑V2 MoE decoder handles complex layouts best, but expect higher memory/latency.
- **Deploying to CPU-only nodes, 16 GB laptops, or latency-sensitive services?** Choose **PaddleOCR‑VL**. Its dense Ernie decoder (18 layers, hidden 1024) activates fewer parameters per token and keeps memory under 10 GB while staying close in quality on most docs.

## Why Rust? 💡

The original DeepSeek-OCR ships as a Python + Transformers stack—powerful, but hefty to deploy and awkward to embed. Rewriting the pipeline in Rust gives us:

- Smaller deployable artifacts with zero Python runtime or conda baggage.
- Memory-safe, thread-friendly infrastructure that blends into native Rust backends.
- Unified tooling (CLI + server) running on Candle + Rocket without the Python GIL overhead.
- Drop-in compatibility with OpenAI-style clients while tuned for single-turn OCR prompts.

## Technical Stack ⚙️

- **Candle** for tensor compute, with Metal and CUDA backends and FlashAttention support.
- **Rocket** + async streaming for OpenAI-compatible `/v1/responses` and `/v1/chat/completions`.
- **tokenizers** (upstream DeepSeek release) wrapped by `crates/assets` for deterministic caching via Hugging Face and ModelScope mirrors.
- **Pure Rust vision/prompt pipeline** shared by CLI and server to avoid duplicated logic.

## Advantages over the Python Release 🥷

- Faster cold-start on Apple Silicon, lower RSS, and native binary distribution.
- Deterministic dual-source (Hugging Face + ModelScope) asset download + verification built into the workspace.
- Automatic single-turn chat compaction so OCR outputs stay stable even when clients send history.
- Ready-to-use OpenAI compatibility for tools like Open WebUI without adapters.

## Highlights ✨

- **One repo, two entrypoints** – a batteries-included CLI for batch jobs and a Rocket-based server that speaks `/v1/responses` and `/v1/chat/completions`.
- **Works out of the box** – pulls model weights, configs, and tokenizer from whichever of Hugging Face or ModelScope responds fastest on first run.
- **Optimised for Apple Silicon** – optional Metal backend with FP16 execution for real-time OCR on laptops.
- **CUDA (alpha)** – experimental support via `--features cuda` + `--device cuda --dtype f16`; expect rough edges while we finish kernel coverage.
- **Intel MKL (preview)** – faster BLAS on x86 via `--features mkl` (install Intel oneMKL beforehand).
- **OpenAI client compatibility** – drop-in replacement for popular SDKs; the server automatically collapses chat history to the latest user turn for OCR-friendly prompts.

## Quick Start 🏁

### Prerequisites

- Rust 1.78+ (edition 2024 support)
- Git
- Optional: Apple Silicon running macOS 13+ for Metal acceleration
- Optional: CUDA 12.2+ toolkit + driver for experimental NVIDIA GPU acceleration on Linux/Windows
- Optional: Intel oneAPI MKL for preview x86 acceleration (see below)
- (Recommended) Hugging Face account with `HF_TOKEN` when pulling from the `deepseek-ai/DeepSeek-OCR` repo (ModelScope is used automatically when it’s faster/reachable).

### Clone the Workspace

```bash
git clone https://github.com/TimmyOVO/deepseek-ocr.rs.git
cd deepseek-ocr.rs
cargo fetch
```

### Model Assets

The first invocation of the CLI or server downloads the config, tokenizer, and `model-00001-of-000001.safetensors` (~6.3GB) into `DeepSeek-OCR/`. To prefetch manually:

```bash
cargo run -p deepseek-ocr-cli --release -- --help # dev profile is extremely slow; always prefer --release
```

> Always include `--release` when running from source; debug builds on this model are extremely slow.
Set `HF_HOME`/`HF_TOKEN` if you store Hugging Face caches elsewhere (ModelScope downloads land alongside the same asset tree). The full model package is ~6.3GB on disk and typically requires ~13GB of RAM headroom during inference (model + activations).

## Configuration & Overrides 🗂️

The CLI and server share the same configuration. On first launch we create a `config.toml` populated with defaults; later runs reuse it so both entrypoints stay in sync.

| Platform | Config file (default) | Model cache root |
| --- | --- | --- |
| Linux | `~/.config/deepseek-ocr/config.toml` | `~/.cache/deepseek-ocr/models/<id>/…` |
| macOS | `~/Library/Application Support/deepseek-ocr/config.toml` | `~/Library/Caches/deepseek-ocr/models/<id>/…` |
| Windows | `%APPDATA%\deepseek-ocr\config.toml` | `%LOCALAPPDATA%\deepseek-ocr\models\<id>\…` |

- Override the location with `--config /path/to/config.toml` (available on both CLI and server). Missing files are created automatically.
- Each `[models.entries."<id>"]` record can point to custom `config`, `tokenizer`, or `weights` files. When omitted we fall back to the cache directory above and download/update assets as required.
- Runtime values resolve in this order: command-line flags → values stored in `config.toml` → built-in defaults. The HTTP API adds a final layer where request payload fields (for example `max_tokens`) override everything else for that call.

The generated file starts with the defaults below; adjust them to persistently change behaviour:

```toml
[models]
active = "deepseek-ocr"

[models.entries.deepseek-ocr]

[inference]
device = "cpu"
template = "plain"
base_size = 1024
image_size = 640
crop_mode = true
max_new_tokens = 512
use_cache = true

[server]
host = "0.0.0.0"
port = 8000
```

- `[models]` picks the active model and lets you add more entries (each entry can point to its own config/tokenizer/weights).
- `[inference]` controls notebook-friendly defaults shared by the CLI and server (device, template, vision sizing, decoding budget, cache usage).
- `[server]` sets the network binding and the model identifier reported by `/v1/models`.

See `crates/cli/README.md` and `crates/server/README.md` for concise override tables.

## Benchmark Snapshot 📊

Single-request Rust CLI (Accelerate backend on macOS) compared with the reference Python pipeline on the same prompt and image:

| Stage                                             | ref total (ms) | ref avg (ms) | python total | python/ref |
|---------------------------------------------------|----------------|--------------|--------------|------------|
| Decode – Overall (`decode.generate`)              | 30077.840      | 30077.840    | 56554.873    | 1.88x      |
| Decode – Token Loop (`decode.iterative`)          | 26930.216      | 26930.216    | 39227.974    | 1.46x      |
| Decode – Prompt Prefill (`decode.prefill`)        | 3147.337       | 3147.337     | 5759.684     | 1.83x      |
| Prompt – Build Tokens (`prompt.build_tokens`)     | 0.466          | 0.466        | 45.434       | 97.42x     |
| Prompt – Render Template (`prompt.render`)        | 0.005          | 0.005        | 0.019        | 3.52x      |
| Vision – Embed Images (`vision.compute_embeddings`)| 6391.435      | 6391.435     | 3953.459     | 0.62x      |
| Vision – Prepare Inputs (`vision.prepare_inputs`) | 62.524         | 62.524       | 45.438       | 0.73x      |

## Command-Line Interface 🖥️

Build and run directly from the workspace:

```bash
cargo run -p deepseek-ocr-cli --release -- \
  --prompt "<image>\n<|grounding|>Convert this receipt to markdown." \
  --image baselines/sample/images/test.png \
  --device cpu --max-new-tokens 512
```

> Tip: `--release` is required for reasonable throughput; debug builds can be 10x slower.

> macOS tip: append `--features metal` to the `cargo run`/`cargo build` commands to compile with Accelerate + Metal backends.
>
> CUDA tip (Linux/Windows): append `--features cuda` and run with `--device cuda --dtype f16` to target NVIDIA GPUs—feature is still alpha, so be ready for quirks.
>
> Intel MKL preview: install Intel oneMKL, then build with `--features mkl` for faster CPU matmuls on x86.

Install the CLI as a binary:

```bash
cargo install --path crates/cli
deepseek-ocr-cli --help
```

Key flags:

- `--prompt` / `--prompt-file`: text with `<image>` slots
- `--image`: path(s) matching `<image>` placeholders
- `--device` and `--dtype`: choose `metal` + `f16` on Apple Silicon or `cuda` + `f16` on NVIDIA GPUs
- `--max-new-tokens`: decoding budget
- Sampling controls: `--do-sample`, `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`, `--no-repeat-ngram-size`, `--seed`
  - By default decoding stays deterministic (`do_sample=false`, `temperature=0.0`, `no_repeat_ngram_size=20`)
  - To use stochastic sampling set `--do-sample true --temperature 0.8` (and optionally adjust the other knobs)

### Switching Models

The autogenerated `config.toml` now contains two model entries:

- `deepseek-ocr` (default) – the original DeepSeek vision-language stack.
- `paddleocr-vl` – the PaddleOCR-VL 0.9B SigLIP + Ernie release.

Pick which one to load via `--model`:

```bash
deepseek-ocr-cli --model paddleocr-vl --prompt "<image> Summarise"
```

The CLI (and server) will download the matching config/tokenizer/weights from the appropriate repository (`deepseek-ai/DeepSeek-OCR` or `PaddlePaddle/PaddleOCR-VL`) into your cache on first use. You can still override paths with `--model-config`, `--tokenizer`, or `--weights` if you maintain local fine-tunes.

## HTTP Server ☁️

Launch an OpenAI-compatible endpoint:

```bash
cargo run -p deepseek-ocr-server --release -- \
  --host 0.0.0.0 --port 8000 \
  --device cpu --max-new-tokens 512
```

> Keep `--release` on the server as well; the debug profile is far too slow for inference workloads.
> macOS tip: add `--features metal` to the `cargo run -p deepseek-ocr-server` command when you want the server binary to link against Accelerate + Metal (and pair it with `--device metal` at runtime).
>
> CUDA tip: add `--features cuda` and start the server with `--device cuda --dtype f16` to offload inference to NVIDIA GPUs (alpha-quality support).
>
> Intel MKL preview: install Intel oneMKL before building with `--features mkl` to accelerate CPU workloads on x86.

Notes:

- Use `data:` URLs or remote `http(s)` links; local paths are rejected.
- The server collapses multi-turn chat inputs to the latest user message to keep prompts OCR-friendly.
- Works out of the box with tools such as [Open WebUI](https://github.com/open-webui/open-webui) or any OpenAI-compatible client—just point the base URL to your server (`http://localhost:8000/v1`) and select either the `deepseek-ocr` or `paddleocr-vl` model ID exposed in `/v1/models`.
- Adjust the request body limit with Rocket config if you routinely send large images.

![Open WebUI connected to deepseek-ocr.rs](./baselines/sample_1.png)

## GPU Acceleration ⚡

- **Metal (macOS 13+ Apple Silicon)** – pass `--device metal --dtype f16` and build binaries with `--features metal` so Candle links against Accelerate + Metal.
- **CUDA (alpha, NVIDIA GPUs)** – install CUDA 12.2+ toolkits, build with `--features cuda`, and launch the CLI/server with `--device cuda --dtype f16`; still experimental.
- **Intel MKL (preview)** – install Intel oneMKL and build with `--features mkl` to speed up CPU workloads on x86.
- For either backend, prefer release builds (e.g. `cargo build --release -p deepseek-ocr-cli --features metal|cuda`) to maximise throughput.
- Combine GPU runs with `--max-new-tokens` and crop tuning flags to balance latency vs. quality.

## Repository Layout 🗂️

- `crates/core` – shared inference pipeline, model loaders, conversation templates.
- `crates/cli` – command-line frontend (`deepseek-ocr-cli`).
- `crates/server` – Rocket server exposing OpenAI-compatible endpoints.
- `crates/assets` – asset management (configuration, tokenizer, Hugging Face + ModelScope download helpers).
- `baselines/` – reference inputs and outputs for regression testing.

Detailed CLI usage lives in [`crates/cli/README.md`](crates/cli/README.md). The server’s OpenAI-compatible interface is covered in [`crates/server/README.md`](crates/server/README.md).

## Troubleshooting 🛠️

- **Where do assets come from?** – downloads automatically pick between Hugging Face and ModelScope based on latency; the CLI prints the chosen source for each file.
- **Slow first response** – model load and GPU warm-up (Metal/CUDA alpha) happen on the initial request; later runs are faster.
- **Large image rejection** – increase Rocket JSON limits in `crates/server/src/main.rs` or downscale the input.

## Roadmap 🗺️

- ✅ Apple Metal backend with FP16 support and CLI/server parity on macOS.
- ✅ NVIDIA CUDA backend (alpha) – build with `--features cuda`, run with `--device cuda --dtype f16` for Linux/Windows GPUs; polishing in progress.
- 🔄 **Parity polish** – finish projector normalisation + crop tiling alignment; extend intermediate-tensor diff suite beyond the current sample baseline.
- 🔄 **Grounding & streaming** – port the Python post-processing helpers (box extraction, markdown polish) and refine SSE streaming ergonomics.
- 🔄 **Cross-platform acceleration** – continue tuning CUDA kernels, add automatic device detection across CPU/Metal/CUDA, and publish opt-in GPU benchmarks.
- 🔄 **Packaging & Ops** – ship binary releases with deterministic asset checksums, richer logging/metrics, and Helm/docker references for server deploys.
- 🔜 **Structured outputs** – optional JSON schema tools for downstream automation once parity gaps close.

## License 📄

This repository inherits the licenses of its dependencies and the upstream DeepSeek-OCR model. Refer to `DeepSeek-OCR/LICENSE` for model terms and apply the same restrictions to downstream use.
