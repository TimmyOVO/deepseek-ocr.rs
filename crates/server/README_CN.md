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
| `--tokenizer PATH` | 资产默认路径 | 指定自定义分词器路径，默认会自动下载。 |
| `--weights PATH` | 自动探测 | 指定替代模型权重的 safetensor 文件。 |
| `--device` | `cpu` | 推理后端：`cpu`、`metal` 或 `cuda`（预览）。 |
| `--dtype` | 依后端而定 | 精度覆盖，如 `f32`、`f16`、`bf16`。 |
| `--base-size` | `1024` | 传入视觉模块的全局视图分辨率。 |
| `--image-size` | `640` | 启用动态裁剪时的局部分辨率。 |
| `--crop-mode` | `true` | 是否启用动态裁剪（`false` 可关闭）。 |
| `--max-new-tokens` | `512` | 服务端默认的解码上限，可被请求体中的 `max_tokens` 覆盖。 |
| `--host` | `0.0.0.0` | Rocket 绑定的地址。 |
| `--do-sample` | `false` | 是否默认启用采样（请求体可再次覆写）。 |
| `--temperature` | `0.0` | 采样温度，需要与 `--do-sample` 配合且 >0 才生效。 |
| `--top-p` | `1.0` | 核心采样累计概率，仅在采样时生效。 |
| `--top-k` | – | Top-k 截断，同样仅在采样时使用。 |
| `--repetition-penalty` | `1.0` | 重复惩罚系数（>1 会降低重复概率）。 |
| `--no-repeat-ngram-size` | `20` | 全局 n-gram 阻断窗口。 |
| `--seed` | – | 采样随机种子，主要用于调试复现。 |
| `--port` | `8000` | HTTP 监听端口。 |
| `--model-id` | `deepseek-ocr` | `/v1/models` 以及流式响应中返回的模型名。 |

> **截断提示：** 如果客户端响应过早结束，请调大 `--max-new-tokens`（或请求体 `max_tokens`）。只要达到该上限，模型就会停止生成。

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
