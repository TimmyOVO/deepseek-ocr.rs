> For the English version please read [README.md](README.md).

`deepseek-ocr-server` 提供 OpenAI 兼容的 HTTP 接口（`/v1/responses`、`/v1/chat/completions`、`/v1/models`），适合需要流式输出或与 Open WebUI 等工具对接的场景。

```bash
cargo run -p deepseek-ocr-server -- \
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
| `--port` | `8000` | HTTP 监听端口。 |
| `--model-id` | `deepseek-ocr` | `/v1/models` 以及流式响应中返回的模型名。 |

> **截断提示：** 如果客户端响应过早结束，请调大 `--max-new-tokens`（或请求体 `max_tokens`）。只要达到该上限，模型就会停止生成。

## 使用说明

- 使用 GPU 后端（`--device metal` 或 `--device cuda`）时，需要在 `cargo run/build` 时加入对应的 `--features metal` 或 `--features cuda`。
- 服务端会将多轮对话压缩为最近的用户消息，以保持 OCR 友好；推荐单轮请求。
- 想跨机器复用模型资源，首次启动前设置 `HF_HOME` 指向共享缓存目录。
