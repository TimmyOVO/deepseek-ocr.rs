> For the English version please read [README.md](README.md).

`deepseek-ocr-server` 提供 OpenAI 兼容的 HTTP 接口（`/v1/responses`、`/v1/chat/completions`、`/v1/models`），适合需要流式输出或与 Open WebUI 等工具对接的场景。

```bash
cargo run -p deepseek-ocr-server --release -- \
  --host 0.0.0.0 \
  --port 8000 \
  --device cpu \
  --max-new-tokens 512
```

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--tokenizer PATH` | 资产默认路径 | 指定自定义 Tokenizer 路径，默认会自动下载。 |
| `--weights PATH` | 自动探测 | 指定替代模型权重的 safetensor 文件。 |
| `--config PATH` | 平台默认 | 读取或初始化指定路径的配置文件。 |
| `--model ID` | `deepseek-ocr` | 选择要服务的模型条目（`deepseek-ocr`、`paddleocr-vl` 或自定义）。 |
| `--model-config PATH` | 模型默认 | 覆盖所选模型的 JSON 配置路径。 |
| `--device` | `cpu` | 推理后端：`cpu`、`metal` 或 `cuda`（预览）。 |
| `--dtype` | 依后端而定 | 精度覆盖，如 `f32`、`f16`、`bf16`。 |
| `--base-size` | `1024` | 传入视觉模块的全局视图分辨率。 |
| `--image-size` | `640` | 启用 dynamic crop mode 时的局部分辨率（仅 DeepSeek-OCR 生效）。 |
| `--crop-mode` | `true` | 是否启用 dynamic crop mode（仅 DeepSeek-OCR 生效，PaddleOCR-VL 会忽略）。 |
| `--max-new-tokens` | `512` | 服务端默认的 decoding 上限，可被请求体中的 `max_tokens` 覆盖。 |
| `--host` | `0.0.0.0` | Rocket 绑定的地址。 |
| `--do-sample` | `false` | 是否默认启用 sampling（请求体可再次覆写）。 |
| `--temperature` | `0.0` | sampling temperature，需要与 `--do-sample` 配合且 >0 才生效。 |
| `--top-p` | `1.0` | top‑p（nucleus sampling 概率质量），仅在 sampling 时生效。 |
| `--top-k` | – | top‑k 截断，同样仅在 sampling 时使用。 |
| `--repetition-penalty` | `1.0` | repetition penalty（>1 会降低重复概率）。 |
| `--no-repeat-ngram-size` | `20` | 全局 no‑repeat n‑gram size。 |
| `--seed` | – | sampling 随机种子，主要用于调试复现。 |
| `--port` | `8000` | HTTP 监听端口。 |

> **截断提示：** 如果客户端响应过早结束，请调大 `--max-new-tokens`（或请求体 `max_tokens`）。只要达到该上限，模型就会停止生成。

## 模型选择

- `config.toml` 的 `[models.entries]` 定义了所有可用后端（默认包含 `deepseek-ocr`、`paddleocr-vl`）。通过 `--model` 或修改 `[models].active` 可以指定启动时预加载的模型。
- 每个 `/v1/responses`、`/v1/chat/completions` 请求都必须携带 `model` 字段。若请求的模型与当前缓存不同，服务端会先卸载旧模型，再加载对应权重（必要时自动下载），随后执行推理——内存中始终只保留一个模型，因此频繁切换会带来一次性加载开销。
- `/v1/models` 会列出同样的模型 ID，方便 OpenAI 兼容客户端动态发现。

## 配置与覆盖

| 平台 | 配置文件路径 | 权重缓存路径 |
| --- | --- | --- |
| Linux | `~/.config/deepseek-ocr/config.toml` | `~/.cache/deepseek-ocr/models/<id>/model.safetensors` |
| macOS | `~/Library/Application Support/deepseek-ocr/config.toml` | `~/Library/Caches/deepseek-ocr/models/<id>/model.safetensors` |
| Windows | `%APPDATA%\deepseek-ocr/config.toml` | `%LOCALAPPDATA%\deepseek-ocr\models\<id>\model.safetensors` |

- 通过 `--config /path/to/config.toml` 可加载或初始化自定义路径，若文件不存在会写入默认内容。
- 生效顺序为：命令行参数 → `config.toml` → 内置默认值；HTTP 请求体中的字段（如 `max_tokens`）会在该次请求内再次覆盖。资产路径同样遵循此顺序：显式参数 > 配置文件 > 上表所示缓存目录。
- 默认配置（包含推理与服务端段落）可在仓库根部 `README_CN.md` 中查看，根据需要修改即可长期生效。

## 使用说明

- 使用 GPU 后端（`--device metal` 或 `--device cuda`）时，需要在 `cargo run/build` 时加入对应的 `--features metal` 或 `--features cuda`。
- 服务端会将多轮对话压缩为最近的用户消息，以保持 OCR 友好；推荐单轮请求。
- 想跨机器复用模型资源，首次启动前设置 `HF_HOME` 指向共享缓存目录。
