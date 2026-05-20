# deepseek-ocr.rs 🚀

Rust implementation of the DeepSeek-OCR inference stack with a fast CLI and an OpenAI-compatible HTTP server. The workspace packages multiple OCR backends, prompt tooling, and a serving layer so you can build document understanding pipelines that run locally on CPU, Apple Metal, or (alpha) NVIDIA CUDA GPUs.

> 中文文档请看 [README_CN.md](README_CN.md)。  

> Want ready-made binaries? Latest macOS (Metal-enabled) and Windows bundles live in the [build-binaries workflow artifacts](https://github.com/TimmyOVO/deepseek-ocr.rs/actions/workflows/build-binaries.yml). Grab them from the newest green run.

## Choosing a Model 🔬

| Model | Memory footprint* | Best on | When to pick it |
| --- | --- | --- | --- |
| **DeepSeek‑OCR** | **≈6.3GB** FP16 weights, **≈13GB** RAM/VRAM with cache & activations (512-token budget) | Apple Silicon + Metal (FP16), high-VRAM NVIDIA GPUs, 32GB+ RAM desktops | Highest accuracy, SAM+CLIP global/local context, MoE DeepSeek‑V2 decoder (3B params, ~570M active per token). Use when latency is secondary to quality. |
| **PaddleOCR‑VL** | **≈4.7GB** FP16 weights, **≈9GB** RAM/VRAM with cache & activations | 16GB laptops, CPU-only boxes, mid-range GPUs | Dense 0.9B Ernie decoder with SigLIP vision tower. Faster startup, lower memory, great for batch jobs or lightweight deployments. |
| **DotsOCR** | **≈9GB** FP16 weights, but expect **30–50GB** RAM/VRAM for high-res docs due to huge vision tokens | Apple Silicon + Metal BF16, ≥24GB CUDA cards, or 64GB RAM CPU workstations | Unified VLM (DotsVision + Qwen2) that nails layout, reading order, grounding, and multilingual math if you can tolerate the latency and memory bill. |

\*Measured from the default FP16 safetensors. Runtime footprint varies with sequence length.

Guidance:

- **Need maximum fidelity, multi-region reasoning, or already have 16–24GB VRAM?** Use **DeepSeek‑OCR**. The hybrid SAM+CLIP tower plus DeepSeek‑V2 MoE decoder handles complex layouts best, but expect higher memory/latency.
- **Deploying to CPU-only nodes, 16GB laptops, or latency-sensitive services?** Choose **PaddleOCR‑VL**. Its dense Ernie decoder (18 layers, hidden 1024) activates fewer parameters per token and keeps memory under 10GB while staying close in quality on most docs.
- **Chasing reading-order accuracy, layout grounding, or multi-page multilingual PDFs on roomy hardware?** Pick **DotsOCR** with BF16 on Metal/CUDA. Prefill runs around 40–50 tok/s on M-series GPUs but can fall to ~12 tok/s on CPU because of the heavy vision tower.

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
- **CUDA** – NVIDIA GPU acceleration via `--features cuda` + `--device cuda --dtype f16`. VRAM swap (`--vision-offload sequential`) brings DeepSeek‑OCR Q4K down to **~2.3GB VRAM peak** on low-end GPUs (RTX 3050 4GB, GTX 1060, etc.). Full GPU vision mode (`--vision-offload full-gpu`) available for higher-VRAM cards. See the [VRAM swap section](#vram-swap--low-vram-gpus) for details.
- **Intel MKL (preview)** – faster BLAS on x86 via `--features mkl` (install Intel oneMKL beforehand).
- **OpenAI client compatibility** – drop-in replacement for popular SDKs; the server automatically collapses chat history to the latest user turn for OCR-friendly prompts.

## Model Matrix 📦

The workspace exposes three base model IDs plus DSQ-quantized variants for DeepSeek‑OCR, PaddleOCR‑VL, and DotsOCR:

| Model ID | Base Model | Precision | Suggested Use Case |
| --- | --- | --- | --- |
| `deepseek-ocr` | `deepseek-ocr` | FP16 (select via `--dtype`) | Full-fidelity DeepSeek‑OCR stack with SAM+CLIP + MoE decoder; use when you prioritise quality on capable Metal/CUDA/CPU hosts. |
| `deepseek-ocr-q4k` | `deepseek-ocr` | `Q4_K` | Tight VRAM, local deployments, and batch jobs that still want DeepSeek’s SAM+CLIP pipeline. |
| `deepseek-ocr-q6k` | `deepseek-ocr` | `Q6_K` | Day‑to‑day balance of quality and size on mid‑range GPUs. |
| `deepseek-ocr-q8k` | `deepseek-ocr` | `Q8_0` | Stay close to full‑precision quality with manageable memory savings. |
| `paddleocr-vl` | `paddleocr-vl` | FP16 (select via `--dtype`) | Default choice for lighter hardware; 0.9B Ernie + SigLIP tower with strong doc/table OCR and low latency. |
| `paddleocr-vl-q4k` | `paddleocr-vl` | `Q4_K` | Heavily compressed doc/table deployments with aggressive memory budgets. |
| `paddleocr-vl-q6k` | `paddleocr-vl` | `Q6_K` | Common engineering setups; blends accuracy and footprint. |
| `paddleocr-vl-q8k` | `paddleocr-vl` | `Q8_0` | Accuracy‑leaning deployments that still want a smaller footprint than FP16. |
| `dots-ocr` | `dots-ocr` | FP16 / BF16 (via `--dtype`) | DotsVision + Qwen2 VLM for high‑precision layout, reading order, grounding, and multilingual docs; expect high memory (30–50GB on large pages). |
| `dots-ocr-q4k` | `dots-ocr` | `Q4_K` | Sidecar DSQ snapshot over the DotsOCR baseline; reduces weight memory/compute while keeping the heavy vision token profile unchanged. |
| `dots-ocr-q6k` | `dots-ocr` | `Q6_K` | Recommended balance of size and quality when you already accept DotsOCR’s memory footprint but want cheaper weights. |
| `dots-ocr-q8k` | `dots-ocr` | `Q8_0` | Accuracy‑leaning DotsOCR deployment that stays close to FP16/BF16 quality with modest memory savings. |


## Quick Start 🏁

### Prerequisites

- Rust 1.78+ (edition 2024 support)
- Git
- Optional: Apple Silicon running macOS 13+ for Metal acceleration
- Optional: CUDA 12.2+ toolkit + driver for NVIDIA GPU acceleration on Linux/Windows
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
vision_swap = true
patches_per_batch = 2
vision_offload = "auto"
max_new_tokens = 512
use_cache = true

[server]
host = "0.0.0.0"
port = 8000
```

- `[models]` picks the active model and lets you add more entries (each entry can point to its own config/tokenizer/weights).
- `[inference]` controls notebook-friendly defaults shared by the CLI and server (device, template, vision sizing, decoding budget, cache usage, VRAM swap settings).
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
> CUDA tip (Linux/Windows): append `--features cuda` and run with `--device cuda --dtype f16` to target NVIDIA GPUs. Add `--vision-offload sequential` for low-VRAM cards.
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
- VRAM swap flags (CUDA only, DeepSeek‑OCR):
  - `--vision-offload <STRATEGY>` – choose vision offloading mode: `auto` (default), `sequential`, `full-gpu`, `cpu`
  - `--vision-swap <BOOL>` – enable/disable VRAM-aware loading (overridden by `--vision-offload`)
  - `--patches-per-batch <N>` – patch batch size for CUDA processing (default: 2; smaller = less VRAM)

### Switching Models

The autogenerated `config.toml` now lists three entries:

- `deepseek-ocr` (default) – the original DeepSeek vision-language stack.
- `paddleocr-vl` – the PaddleOCR-VL 0.9B SigLIP + Ernie release.
- `dots-ocr` – the Candle port of dots.ocr with DotsVision + Qwen2 (use BF16 on Metal/CUDA if possible; see the release matrix for memory notes).

Pick which one to load via `--model`:

```bash
deepseek-ocr-cli --model paddleocr-vl --prompt "<image> Summarise"
```

The CLI (and server) will download the matching config/tokenizer/weights from the appropriate repository (`deepseek-ai/DeepSeek-OCR`, `PaddlePaddle/PaddleOCR-VL`, or `dots-ocr`) into your cache on first use. You can still override paths with `--model-config`, `--tokenizer`, or `--weights` if you maintain local fine-tunes.

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

![Open WebUI connected to deepseek-ocr.rs](./assets/sample_1.png)

## GPU Acceleration ⚡

- **Metal (macOS 13+ Apple Silicon)** – pass `--device metal --dtype f16` and build binaries with `--features metal` so Candle links against Accelerate + Metal.
- **CUDA (NVIDIA GPUs)** – install CUDA 12.2+ toolkits, build with `--features cuda`, and launch the CLI/server with `--device cuda --dtype f16`. Includes VRAM swap support for low-memory GPUs (RTX 3050 4GB, GTX 1060, etc.) and full-GPU vision mode for larger cards. See [VRAM Swap](#vram-swap--low-vram-gpus) below.
- **Intel MKL (preview)** – install Intel oneMKL and build with `--features mkl` to speed up CPU workloads on x86.
- For either backend, prefer release builds (e.g. `cargo build --release -p deepseek-ocr-cli --features metal|cuda`) to maximise throughput.
- Combine GPU runs with `--max-new-tokens` and crop tuning flags to balance latency vs. quality.

## VRAM Swap & Low-VRAM GPUs 🔄

DeepSeek‑OCR with Q4K quantisation (`deepseek-ocr-q4k`) fits in **~950 MB** on CUDA, but the SAM+CLIP vision models require an additional **~2.1 GB**. On low-VRAM GPUs (≤6 GB), a **VRAM swap** mode loads one vision model at a time:

| Mode | `--vision-offload` | Global View | Patch Crops | Peak VRAM (Q4K) |
|------|-------------------|-------------|-------------|-----------------|
| **Auto** (default) | `auto` | Auto-detects: uses Sequential on <6 GB | | ~2.3 GB |
| **Sequential** | `sequential` | CPU (always) | CUDA (chunked) | ~2.3 GB |
| **Full GPU** | `full-gpu` | CUDA | CUDA (chunked) | ~3.2 GB |
| **CPU only** | `cpu` | CPU | CPU | ~1 GB |

**Sequential mode** runs the global view on CPU while patch crops process on CUDA in parallel (scoped threads), giving the best VRAM/speed trade-off on tight GPUs.

**Full GPU mode** keeps all vision on CUDA: loads SAM → processes all images → drops SAM → loads CLIP → processes with cached SAM outputs. Offers faster vision prefill when VRAM allows (~3.2 GB peak).

Fine-tune VRAM vs. throughput with `--patches-per-batch`:
```bash
# Minimum VRAM (processes one patch at a time on CUDA)
cargo run --release --features cuda -- --device cuda --dtype f16 --model deepseek-ocr-q4k \
  --vision-offload sequential --patches-per-batch 1 --prompt "<image> OCR" --image doc.png

# Faster (batches 4 patches per CUDA forward)
cargo run --release --features cuda -- --device cuda --dtype f16 --model deepseek-ocr-q4k \
  --vision-offload sequential --patches-per-batch 4 --prompt "<image> OCR" --image doc.png
```

Disable swap entirely (force CPU vision) when you have enough RAM:
```bash
cargo run --release --features cuda -- --device cuda --dtype f16 --model deepseek-ocr-q4k \
  --vision-offload cpu --prompt "<image> OCR" --image doc.png
```

> **Note:** VRAM swap is only relevant for the DeepSeek‑OCR vision pipeline (SAM+CLIP). PaddleOCR‑VL and DotsOCR do not benefit since they use lighter or different vision towers.

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
- ✅ NVIDIA CUDA backend – build with `--features cuda`, run with `--device cuda --dtype f16` for Linux/Windows GPUs. VRAM swap brings DeepSeek‑OCR Q4K down to ~2.3 GB on RTX 3050 4 GB.
- ✅ **VRAM Swap** – sequential vision offload for low-VRAM CUDA GPUs via `--vision-offload sequential`, full-GPU mode via `--vision-offload full-gpu`, CPU-only mode via `--vision-offload cpu`.
- 🔄 **Parity polish** – finish projector normalisation + crop tiling alignment; extend intermediate-tensor diff suite beyond the current sample baseline.
- 🔄 **Grounding & streaming** – port the Python post-processing helpers (box extraction, markdown polish) and refine SSE streaming ergonomics.
- 🔄 **Cross-platform acceleration** – continue tuning CUDA kernels, add automatic device detection across CPU/Metal/CUDA, and publish opt-in GPU benchmarks.
- 🔄 **Packaging & Ops** – ship binary releases with deterministic asset checksums, richer logging/metrics, and Helm/docker references for server deploys.
- 🔜 **Structured outputs** – optional JSON schema tools for downstream automation once parity gaps close.

## License 📄

This repository inherits the licenses of its dependencies and the upstream DeepSeek-OCR model. Refer to `DeepSeek-OCR/LICENSE` for model terms and apply the same restrictions to downstream use.
