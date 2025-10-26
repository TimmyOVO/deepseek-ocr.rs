> For the English version please read [README.md](README.md).

`deepseek-ocr-cli` 是完整推理流程的终端入口。它接收包含 `<image>` 占位符的文本提示，将图像编码为视觉特征，并自回归生成 Markdown 输出。可以直接在工作区运行，也可以 `cargo install --path crates/cli` 后独立调用。

```bash
cargo run -p deepseek-ocr-cli -- \
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
| `--tokenizer PATH` | 资产默认路径 | 指定自定义分词器路径；默认自动下载并缓存。 |
| `--weights PATH` | 自动探测 | 指定模型权重文件，覆盖默认的 safetensor。 |
| `--device` | `cpu` | 执行后端：`cpu`、`metal` 或 `cuda`（测试阶段）。 |
| `--dtype` | 取决于后端 | 数值精度覆盖选项，如 `f32`、`f16`、`bf16` 等。 |
| `--base-size` | `1024` | 传入视觉模块的全局视图分辨率。 |
| `--image-size` | `640` | 动态裁剪启用时的局部分辨率。 |
| `--crop-mode` | `true` | 是否启用动态裁剪（传 `false` 可关闭）。 |
| `--max-new-tokens` | `512` | 解码阶段允许输出的最大 token 数。 |
| `--no-cache` | `false` | 禁用解码 KV 缓存，仅在调试时使用。 |

> **重要提醒：** 如果生成的 Markdown 被提前截断，请调大 `--max-new-tokens`。模型在达到该上限后会立刻停止，即便尚未完成回答。

## 使用提示

- 模板渲染后 `<image>` 的数量必须与传入的 `--image` 参数数量一致，运行时会自动校验。
- 使用 GPU 后端（`--device metal` 或 `--device cuda`）时，需要在 `cargo run/build` 时配合 `--features metal` 或 `--features cuda` 编译选项。
- 想在多台机器共享模型资源，可在首次运行前设置 `HF_HOME` 指向共享的缓存目录。
