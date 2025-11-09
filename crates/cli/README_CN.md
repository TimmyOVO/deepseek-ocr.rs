> For the English version please read [README.md](README.md).

`deepseek-ocr-cli` 是完整推理流程的终端入口。它接收包含 `<image>` 占位符的文本提示，将图像编码为视觉特征，并自回归生成 Markdown 输出。可以直接在工作区运行，也可以 `cargo install --path crates/cli` 后独立调用。

```bash
cargo run -p deepseek-ocr-cli --release -- \
  --prompt "<image>\n<|grounding|>Convert this Markdown" \
  --image baselines/sample/images/test.png \
  --device cpu --max-new-tokens 512
```

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--prompt` | – | 内联文本提示，使用 `<image>` 标记图片位置。 |
| `--prompt-file` | – | 含提示词的 UTF-8 文件；提供后会覆盖 `--prompt`。 |
| `--template` | `plain` | 会话模板，可选 `plain`、`deepseek`、`deepseekv2`、`alignment`。 |
| `--image PATH` | – | 与 `<image>` 匹配的图片路径，按出现顺序重复传入该参数。 |
| `--config PATH` | 平台默认 | 指定配置文件路径（若不存在会自动生成）。 |
| `--model ID` | `deepseek-ocr` | 选择要加载的模型条目（如 `deepseek-ocr`、`paddleocr-vl` 或自定义 ID）。 |
| `--model-config PATH` | 模型默认 | 覆盖所选模型的 JSON 配置路径。 |
| `--tokenizer PATH` | 资产默认路径 | 指定自定义 Tokenizer 路径；默认自动下载并缓存。 |
| `--weights PATH` | 自动探测 | 指定模型权重文件，覆盖默认的 safetensor。 |
| `--device` | `cpu` | 执行后端：`cpu`、`metal` 或 `cuda`（测试阶段）。 |
| `--dtype` | 取决于后端 | 数值精度覆盖选项，如 `f32`、`f16`、`bf16` 等。 |
| `--base-size` | `1024` | 传入视觉模块的全局视图分辨率。 |
| `--image-size` | `640` | dynamic crop mode 启用时的局部分辨率（仅 DeepSeek-OCR 生效）。 |
| `--crop-mode` | `true` | 是否启用 dynamic crop mode（仅 DeepSeek-OCR 生效，PaddleOCR-VL 会忽略）。 |
| `--max-new-tokens` | `512` | 解码阶段允许输出的最大 token 数。 |
| `--no-cache` | `false` | 禁用 decoder KV Cache，仅在调试时使用。 |
| `--do-sample` | `false` | 是否启用 sampling（需搭配 `--temperature > 0`）。 |
| `--temperature` | `0.0` | sampling temperature，越大越随机。 |
| `--top-p` | `1.0` | top‑p（nucleus sampling 概率质量），仅在 sampling 时生效。 |
| `--top-k` | – | top‑k 截断，配合 sampling 使用。 |
| `--repetition-penalty` | `1.0` | repetition penalty（>1 会降低重复概率）。 |
| `--no-repeat-ngram-size` | `20` | no‑repeat n‑gram size，生成时始终生效。 |
| `--seed` | – | 随机种子，便于复现 sampling 结果。 |

> **重要提醒：** 如果生成的 Markdown 被提前截断，请调大 `--max-new-tokens`。模型在达到该上限后会立刻停止，即便尚未完成回答。

## 模型选择

CLI 通过配置文件中的“模型注册表”支持多种推理引擎：

- 默认内置两条目：
  - `deepseek-ocr`（DeepSeek‑OCR；默认权重 `DeepSeek-OCR/model-00001-of-000001.safetensors`）
  - `paddleocr-vl`（PaddleOCR‑VL；默认权重 `PaddleOCR-VL/model.safetensors`）

可用命令行或配置切换所用模型：

```bash
# 本次运行切换为 PaddleOCR‑VL
cargo run -p deepseek-ocr-cli --release -- \
  --model paddleocr-vl \
  --prompt "<image> 提取表格项" \
  --image baselines/fixtures/paddleocr_vl/fixture_image.png

# 或在配置文件中持久化该选择（TOML 片段）
[models]
active = "paddleocr-vl"

[models.entries.deepseek-ocr]
kind = "deepseek"

[models.entries.paddleocr-vl]
kind = "paddle_ocr_vl"
```

### 配置与覆盖

| 平台 | 配置文件路径 | 权重缓存路径 |
| --- | --- | --- |
| Linux | `~/.config/deepseek-ocr/config.toml` | `~/.cache/deepseek-ocr/models/<id>/model.safetensors` |
| macOS | `~/Library/Application Support/deepseek-ocr/config.toml` | `~/Library/Caches/deepseek-ocr/models/<id>/model.safetensors` |
| Windows | `%APPDATA%\deepseek-ocr\config.toml` | `%LOCALAPPDATA%\deepseek-ocr\models\<id>\model.safetensors` |

- 通过 `--config /path/to/config.toml` 可切换或初始化自定义路径；若文件不存在会自动填入默认值。
- 参数生效顺序为：命令行参数 → `config.toml` → 内置默认值。资产路径同样遵循该顺序：显式的 `--weights`/`--tokenizer` 会覆盖配置文件，若都未指定则使用上表所列缓存目录。
- 默认文件内容可在仓库根目录 `README_CN.md` 中查看，修改对应段落即可长期改变设备、模板、token 上限或 server 监听配置。

## 使用提示

- 模板渲染后 `<image>` 的数量必须与传入的 `--image` 参数数量一致，运行时会自动校验。
- 使用 GPU 后端（`--device metal` 或 `--device cuda`）时，需要在 `cargo run/build` 时配合 `--features metal` 或 `--features cuda` 编译选项。
- 想在多台机器共享模型资源，可在首次运行前设置 `HF_HOME` 指向共享的缓存目录。
