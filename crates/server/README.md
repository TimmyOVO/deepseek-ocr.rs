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
| `--device` | `cpu` | Backend for inference: `cpu`, `metal`, or `cuda` (preview). |
| `--dtype` | backend default | Numeric precision override (`f32`, `f16`, `bf16`, …). |
| `--base-size` | `1024` | Global canvas resolution for the vision stack. |
| `--image-size` | `640` | Local crop size when dynamic tiling is enabled. |
| `--crop-mode` | `true` | Enables dynamic crop mode (`false` to disable). |
| `--max-new-tokens` | `512` | Default decoding budget applied to incoming requests. |
| `--host` | `0.0.0.0` | Address Rocket binds to. |
| `--port` | `8000` | TCP port for the HTTP server. |
| `--model-id` | `deepseek-ocr` | Model name returned by `/v1/models` and streamed responses. |

> **Truncation reminder:** If client responses appear cut off, raise `--max-new-tokens` (or the per-request `max_tokens` body field). The server stops generation once the configured budget is consumed.

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
