> 中文文档请见 [README_CN.md](README_CN.md)。

The `deepseek-ocr-server` binary exposes the model behind an OpenAI-compatible HTTP API (`/v1/responses`, `/v1/chat/completions`, `/v1/models`). Use it when you need streaming output or to integrate with tools such as Open WebUI.

```bash
cargo run -p deepseek-ocr-server --release -- \
  --host 0.0.0.0 \
  --port 8000 \
  --device cpu \
  --max-new-tokens 512
```

## Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--tokenizer PATH` | assets default | Override tokenizer path; otherwise downloaded automatically. |
| `--weights PATH` | auto-detected | Alternate safetensor checkpoint for the model. |
| `--config PATH` | platform default | Load or bootstrap an alternate config file. |
| `--model ID` | `deepseek-ocr` | Select a configured model entry to serve (`deepseek-ocr`, `paddleocr-vl`, or custom). |
| `--model-config PATH` | per-model default | Override the JSON config for the selected model. |
| `--device` | `cpu` | Backend for inference: `cpu`, `metal`, or `cuda` (preview). |
| `--dtype` | backend default | Numeric precision override (`f32`, `f16`, `bf16`, …). |
| `--base-size` | `1024` | Global canvas resolution for the vision stack. |
| `--image-size` | `640` | Local crop size when dynamic tiling is enabled (DeepSeek-OCR only). |
| `--crop-mode` | `true` | Enables dynamic crop mode (DeepSeek-OCR only; ignored for PaddleOCR-VL). |
| `--max-new-tokens` | `512` | Default decoding budget applied to incoming requests. |
| `--host` | `0.0.0.0` | Address Rocket binds to. |
| `--do-sample` | `false` | Enable sampling for all requests unless overridden per-call. |
| `--temperature` | `0.0` | Sampling temperature; must be >0 alongside `--do-sample`. |
| `--top-p` | `1.0` | Nucleus sampling mass (fraction of probability kept). |
| `--top-k` | – | Top-k cutoff applied during sampling. |
| `--repetition-penalty` | `1.0` | Token repetition penalty (>1 discourages repeats). |
| `--no-repeat-ngram-size` | `20` | N-gram blocking window enforced during decoding. |
| `--seed` | – | RNG seed for sampling (mainly for debugging). |
| `--port` | `8000` | TCP port for the HTTP server. |

> **Truncation reminder:** If client responses appear cut off, raise `--max-new-tokens` (or the per-request `max_tokens` body field). The server stops generation once the configured budget is consumed.

## Model selection

- `[models.entries]` in `config.toml` enumerates every supported backend (defaults: `deepseek-ocr`, `paddleocr-vl`). Use the `--model` CLI flag or edit `[models].active` to decide which one is preloaded at startup.
- Every `/v1/responses` or `/v1/chat/completions` request must set the `model` field. When it differs from the currently cached backend, the server will unload the old engine, load the requested model (downloading assets if needed), and then process the call. Only one model stays in memory at a time, so rapid switching can incur reload latency.
- `/v1/models` lists the same IDs so OpenAI-compatible clients can discover them dynamically.

## Configuration & Overrides

| Platform | Config path | Weights cache path |
| --- | --- | --- |
| Linux | `~/.config/deepseek-ocr/config.toml` | `~/.cache/deepseek-ocr/models/<id>/model.safetensors` |
| macOS | `~/Library/Application Support/deepseek-ocr/config.toml` | `~/Library/Caches/deepseek-ocr/models/<id>/model.safetensors` |
| Windows | `%APPDATA%\deepseek-ocr\config.toml` | `%LOCALAPPDATA%\deepseek-ocr\models\<id>\model.safetensors` |

- Use `--config /path/to/config.toml` to load or bootstrap a custom file. Missing files are generated with defaults.
- Effective values resolve in this order: CLI/server flags → entries in `config.toml` → baked-in defaults. For per-request behaviour the JSON payload wins last (for example `max_tokens` overrides both the CLI flag and config setting). Asset paths behave the same way; explicit flags beat config entries which beat the auto-managed cache paths listed above.
- The default TOML layout (including inference and server sections) is documented in the workspace `README.md`; tweak it to persistently change bindings or token budgets.

## Usage Notes

- GPU backends (`--device metal` or `--device cuda`) require compiling with `--features metal` or `--features cuda` respectively.
- The server collapses chat history to the latest user message so prompts stay OCR-focused. Supply single-turn requests for best results.
- For assets shared across machines, set `HF_HOME` before the first launch to reuse cached downloads.
