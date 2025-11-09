> 中文文档请见 [README_CN.md](README_CN.md)。

The `deepseek-ocr-cli` binary drives the full multimodal pipeline from the terminal. It accepts a text prompt with `<image>` placeholders, projects one or more images into vision embeddings, and then autoregressively generates markdown output. Run it directly from the workspace or after `cargo install --path crates/cli`.

```bash
cargo run -p deepseek-ocr-cli --release -- \
  --prompt "<image>\n<|grounding|>Convert this Markdown." \
  --image baselines/sample/images/test.png \
  --device cpu --max-new-tokens 512
```

### Arguments

| Flag | Default | Description |
| --- | --- | --- |
| `--prompt` | – | Inline text with `<image>` markers. |
| `--prompt-file` | – | UTF-8 file containing the prompt; overrides `--prompt`. |
| `--template` | `plain` | Conversation template (`plain`, `deepseek`, `deepseekv2`, `alignment`). |
| `--image PATH` | – | Image path for each `<image>` token, specified in order. Repeat the flag for multiple images. |
| `--config PATH` | platform default | Read/initialise an alternate config file. |
| `--model ID` | `deepseek-ocr` | Select a configured model entry (`deepseek-ocr`, `paddleocr-vl`, or a custom ID). |
| `--model-config PATH` | per-model default | Override the JSON config for the selected model. |
| `--tokenizer PATH` | assets default | Override tokenizer location; downloaded automatically when omitted. |
| `--weights PATH` | auto-detected | Use custom model weights instead of the default safetensor. |
| `--device` | `cpu` | Execution backend: `cpu`, `metal`, or `cuda` (alpha). |
| `--dtype` | backend default | Override numeric precision (`f32`, `f16`, `bf16`, …). |
| `--base-size` | `1024` | Global view resolution supplied to the vision stack. |
| `--image-size` | `640` | Local crop resolution when dynamic tiling is enabled (DeepSeek-OCR only). |
| `--crop-mode` | `true` | Toggle dynamic crop sampling (DeepSeek-OCR only; ignored by PaddleOCR-VL). |
| `--max-new-tokens` | `512` | Maximum number of tokens generated during decoding. |
| `--no-cache` | `false` | Disable the decoder KV-cache. Helpful for debugging only. |
| `--do-sample` | `false` | Enable sampling (requires `--temperature > 0`). |
| `--temperature` | `0.0` | Softmax temperature used when sampling is enabled. |
| `--top-p` | `1.0` | Nucleus sampling mass; ignored unless sampling. |
| `--top-k` | – | Top-k cutoff during sampling. |
| `--repetition-penalty` | `1.0` | Penalise previously generated tokens (>1 discourages repeats). |
| `--no-repeat-ngram-size` | `20` | N-gram blocking window applied to every decode step. |
| `--seed` | – | RNG seed for reproducible sampling runs. |

> **Heads-up:** If the final markdown appears truncated, increase `--max-new-tokens`. The model stops once it has emitted the configured number of tokens even if the prompt is unfinished.

### Model selection

This CLI supports multiple inference engines through a model registry in `config.toml`.

- Default entries:
  - `deepseek-ocr` (DeepSeek-OCR; default weights `DeepSeek-OCR/model-00001-of-000001.safetensors`)
  - `paddleocr-vl` (PaddleOCR‑VL; default weights `PaddleOCR-VL/model.safetensors`)

Pick the active model via CLI or config:

```bash
# Use PaddleOCR-VL for this run
cargo run -p deepseek-ocr-cli --release -- \
  --model paddleocr-vl \
  --prompt "<image> Extract the items." \
  --image baselines/fixtures/paddleocr_vl/fixture_image.png

# Or persist it in the config (TOML excerpt)
[models]
active = "paddleocr-vl"

[models.entries.deepseek-ocr]
kind = "deepseek"

[models.entries.paddleocr-vl]
kind = "paddle_ocr_vl"
```

### Configuration & Overrides

| Platform | Config path | Weights cache path |
| --- | --- | --- |
| Linux | `~/.config/deepseek-ocr/config.toml` | `~/.cache/deepseek-ocr/models/<id>/model.safetensors` |
| macOS | `~/Library/Application Support/deepseek-ocr/config.toml` | `~/Library/Caches/deepseek-ocr/models/<id>/model.safetensors` |
| Windows | `%APPDATA%\deepseek-ocr\config.toml` | `%LOCALAPPDATA%\deepseek-ocr\models\<id>\model.safetensors` |

- Pass `--config /path/to/config.toml` to read or bootstrap an alternate file (created with defaults if missing).
- Runtime values resolve in this order: CLI flags → values in `config.toml` → baked-in defaults. Asset paths behave the same way: explicit flags beat config entries which beat the cache locations listed above.
- The generated file starts with the defaults shown in the workspace root `README.md`; edit them to persistently change devices, templates, token budgets, or server bindings.

### Additional Tips

- Match the number of `--image` arguments to the `<image>` tokens after the template is rendered. The CLI validates this at runtime.
- GPU backends (`--device metal` or `--device cuda`) require compiling with the corresponding Cargo feature (`--features metal` / `--features cuda`).
- To reuse assets across machines, point `HF_HOME` to a shared cache before the first run.
