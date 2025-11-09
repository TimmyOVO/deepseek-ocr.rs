# deepseek-ocr.rs 🚀

Rust 实现的 DeepSeek-OCR 推理栈，提供快速 CLI 与 OpenAI 兼容的 HTTP Server，统一打包多种 OCR 后端、视觉输入预处理、提示词工具与服务端能力，方便在本地 CPU、Apple Metal 或 NVIDIA CUDA GPU(alpha) 上构建文档理解工作流。

> 英文文档请参见 [README.md](README.md)。  

> 想直接下载可执行文件？访问 [Github Actions](https://github.com/TimmyOVO/deepseek-ocr.rs/actions/workflows/build-binaries.yml)，下载最新一次成功运行生成的 macOS（含 Metal）或 Windows 压缩包。

## 模型选择指南 🔬

| 模型 | 内存占用* | 最佳硬件 | 适用场景 |
| --- | --- | --- | --- |
| **DeepSeek‑OCR** | **≈6.3 GB** FP16 权重，含激活/缓存约 **13 GB**（512 token） | Apple Silicon + Metal、24 GB VRAM NVIDIA、32 GB+ RAM 桌面 | 追求最高准确率、多视角文档、对延迟不敏感。SAM+CLIP 视觉 + DeepSeek‑V2 MoE（3 B 参数，单 token 激活 ≈570 M）。 |
| **PaddleOCR‑VL** | **≈4.7 GB** FP16 权重，含激活/缓存约 **9 GB** | 16 GB 笔电、CPU-only 节点、中端 GPU | 更快冷启动，dense Ernie decoder（0.9 B）+ SigLIP 视觉，适合批量作业与轻量部署。 |

\*默认 FP16 safetensors 容量；实际资源与序列长度、是否启用 KV Cache 相关。

选择建议：

- **有 16–24 GB 以上 VRAM / RAM、追求极致质量？** 选 **DeepSeek‑OCR**，SAM+CLIP 全局+局部视野、DeepSeek‑V2 MoE 解码能在复杂版式中保持更高还原度，但代价是更大的显存和更高延迟。
- **硬件预算有限或需要低延迟 / 高吞吐？** 选 **PaddleOCR‑VL**，SigLIP + dense Ernie（18 层、hidden 1024）在 10 GB 以内即可运行，CPU 模式也更易部署。

## 为什么选择 Rust？💡

官方 DeepSeek-OCR 依赖 Python + Transformers，部署体积大、依赖多，嵌入原生系统成本高。Rust 重写后的优势：

- 无需 Python 运行时或 conda，产物更小、更易嵌入。
- 内存安全、线程友好，可直接融入现有 Rust 服务。
- CLI 与 Server 共用一套核心逻辑，避免重复维护。
- 依旧兼容 OpenAI 客户端，同时聚焦单轮 OCR 场景确保输出稳定。

## 技术栈 ⚙️

- **Candle**：Rust 深度学习框架，支持 Metal/CUDA（alpha） 与 FlashAttention。
- **Rocket**：异步 HTTP 框架，提供 `/v1/responses`、`/v1/chat/completions` 等 OpenAI 兼容路由。
- **tokenizers**：上游模型提供的 Tokenizer，通过 `crates/assets` 在 Hugging Face / ModelScope 镜像间缓存与校验。
- **纯 Rust 视觉/Prompt 管线**：CLI 与 Server 复用，减少重复逻辑。

## 相比 Python 实现的优势 🥷

- Apple Silicon 冷启动更快、内存占用更低，且提供原生二进制分发。
- 资产下载/校验由 Rust crate 统一托管，可在 Hugging Face 与 ModelScope 之间自动切换。
- 自动折叠多轮会话，仅保留最近一次 user 提示，确保 OCR 场景稳定。
- 与 Open WebUI 等 OpenAI 客户端“即插即用”，无需额外适配层。

## 项目亮点 ✨

- **一套代码，两种入口**：批处理友好的 CLI 与兼容 `/v1/responses`、`/v1/chat/completions` 的 Rocket Server。
- **开箱即用**：首次运行自动从 Hugging Face 或 ModelScope（取决于实时延迟）拉取配置、Tokenizer 与权重。
- **Apple Silicon 友好**：Metal + FP16 加速让笔记本也能实时 OCR。
- **NVIDIA GPU（α 测试）**：构建时附加 `--features cuda` 并以 `--device cuda --dtype f16` 运行，可在 Linux/Windows 上尝鲜 CUDA 加速。
- **Intel MKL（预览）**：安装 Intel oneMKL 后，构建时附加 `--features mkl` 以提升 x86 CPU 上的矩阵运算速度。
- **OpenAI 客户端即插即用**：Server 端自动折叠多轮对话，只保留最新 user 指令，避免 OCR 模型被多轮上下文干扰。

## 快速上手 🏁

### 环境要求

- Rust 1.78+（支持 2024 Edition）
- Git
- 可选：macOS 13+ 的 Apple Silicon（用于 Metal）
- 可选：Linux/Windows 的 NVIDIA GPU（需 CUDA 12.2+ 工具链与驱动，当前为alpha阶段）
- 可选：x86 平台安装 Intel oneMKL（预览），用于提升 CPU 推理性能
- 推荐：配置 `HF_TOKEN` 访问 Hugging Face `deepseek-ai/DeepSeek-OCR`（若该源不可用会自动切换 ModelScope）

### 克隆仓库

```bash
git clone https://github.com/TimmyOVO/deepseek-ocr.rs.git
cd deepseek-ocr.rs
cargo fetch
```

### 模型资源

第一次运行 CLI 或 Server 会把配置、tokenizer 及 ~6.3GB 的 `model-00001-of-000001.safetensors` 下载到 `DeepSeek-OCR/`。也可以手动触发：

```bash
cargo run -p deepseek-ocr-cli --release -- --help # dev profile 极慢，建议始终加 --release
```

若自定义缓存目录，请设置 `HF_HOME` 或导出 `HF_TOKEN`。完整模型约 6.3GB，推理时需预留 ~13GB 内存（模型 + 激活）。

### 预构建产物

不想自己编译？点这里 [Github Actions](https://github.com/TimmyOVO/deepseek-ocr.rs/actions/workflows/build-binaries.yml) 里产出 macOS（含 Metal）和 Windows 压缩包。登录 GitHub，打开最新一次绿色运行，下载 `deepseek-ocr-macos` 或 `deepseek-ocr-windows` 即可。

## 配置与优先级 🗂️

CLI 与 Server 共享同一份配置。首次启动会在系统配置目录生成带默认值的 `config.toml`，后续运行都会沿用该文件确保两个入口保持一致。

| 平台 | 默认配置文件 | 模型缓存目录 |
| --- | --- | --- |
| Linux | `~/.config/deepseek-ocr/config.toml` | `~/.cache/deepseek-ocr/models/<id>/…` |
| macOS | `~/Library/Application Support/deepseek-ocr/config.toml` | `~/Library/Caches/deepseek-ocr/models/<id>/…` |
| Windows | `%APPDATA%\deepseek-ocr\config.toml` | `%LOCALAPPDATA%\deepseek-ocr\models\<id>\…` |

- 可通过 `--config /path/to/config.toml`（CLI/Server 通用）自定义路径；当文件不存在时会自动创建并写入默认内容。
- 默认的 `config.toml` 已包含 `deepseek-ocr`（默认）与 `paddleocr-vl` 两个模型条目，可通过 `--model paddleocr-vl`（或修改 `[models].active`）在 DeepSeek 与 PaddleOCR-VL 之间即时切换。
- 需要自定义资源位置时，可在对应 `models.entries.<id>` 下设置 `config`/`tokenizer`/`weights`，或直接在运行时使用 `--model-config`、`--tokenizer`、`--weights` 覆盖。
- `config.toml` 中的 `[models.entries."<id>"]` 节点允许为不同模型指定独立的 `config`、`tokenizer`、`weights` 路径；若留空则使用上表所示缓存目录并按需下载。
- 参数覆盖顺序为：命令行参数 → `config.toml` → 内置默认值。HTTP API 请求体中的字段（例如 `max_tokens`）会在该次调用中继续覆盖前述设置。

默认配置文件内容如下，可根据需要修改后长期生效：

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

- `[models]` 用于指定当前激活的模型以及额外的模型条目（每个条目都可以指向各自的配置、Tokenizer 与权重文件）。
- `[inference]` 提供 CLI 与 Server 共用的推理默认值（设备、模板、视觉分辨率、生成长度与缓存策略）。
- `[server]` 决定网络监听地址以及 `/v1/models` 返回的模型名。

更多覆盖项详见 `crates/cli/README_CN.md` 与 `crates/server/README_CN.md`。

## 基准对比 📊

下表展示在同一张图像与提示词下，启用了 Accelerate 的 Rust CLI（单次请求）与 Python 参考实现的性能表现：

| 阶段（Stage）                                     | ref total (ms) | ref avg (ms) | python total | python/ref |
|--------------------------------------------------|----------------|--------------|--------------|------------|
| Decode – Overall (`decode.generate`)             | 30077.840      | 30077.840    | 56554.873    | 1.88x      |
| Decode – Token Loop (`decode.iterative`)         | 26930.216      | 26930.216    | 39227.974    | 1.46x      |
| Decode – Prompt Prefill (`decode.prefill`)       | 3147.337       | 3147.337     | 5759.684     | 1.83x      |
| Prompt – Build Tokens (`prompt.build_tokens`)    | 0.466          | 0.466        | 45.434       | 97.42x     |
| Prompt – Render Template (`prompt.render`)       | 0.005          | 0.005        | 0.019        | 3.52x      |
| Vision – Embed Images (`vision.compute_embeddings`)| 6391.435     | 6391.435     | 3953.459     | 0.62x      |
| Vision – Prepare Inputs (`vision.prepare_inputs`)| 62.524         | 62.524       | 45.438       | 0.73x      |

## 命令行工具 🖥️

直接运行：

```bash
cargo run -p deepseek-ocr-cli --release -- \
  --prompt "<image>\n<|grounding|>Convert this receipt to markdown." \
  --image baselines/sample/images/test.png \
  --device cpu --max-new-tokens 512
```

> macOS 用户可以在 `cargo run`/`cargo build` 命令后附加 `--features metal` 以启用 Accelerate + Metal 后端。
>
> Linux/Windows 用户：附加 `--features cuda` 并在运行参数中加入 `--device cuda --dtype f16`，即可使用 NVIDIA GPU 加速。
>
> Intel MKL 预览：先安装 Intel oneMKL，构建时附加 `--features mkl`，可在 x86 CPU 上取得更高的矩阵运算性能。

安装成全局二进制：

```bash
cargo install --path crates/cli
deepseek-ocr-cli --help
```

常用参数：

- `--prompt` / `--prompt-file`：包含 `<image>` 占位符的提示词
- `--image`：与 `<image>` 数量一致的图片路径
- `--device` / `--dtype`：macOS 建议 `--device metal --dtype f16`，NVIDIA 用户使用 `--device cuda --dtype f16`
- `--max-new-tokens`：生成长度上限
- Sampling 相关：`--do-sample`、`--temperature`、`--top-p`、`--top-k`、`--repetition-penalty`、`--no-repeat-ngram-size`、`--seed`
  - 默认保持确定性输出（`do_sample=false`、`temperature=0.0`、`no_repeat_ngram_size=20`）
  - 若需要随机 sampling，请显式指定 `--do-sample true --temperature 0.8`，并按需调整其他参数

## HTTP Server ☁️

启动 OpenAI 兼容服务：

```bash
cargo run -p deepseek-ocr-server --release -- \
  --host 0.0.0.0 --port 8000 \
  --device cpu --max-new-tokens 512
```

> 如果要在 macOS 上启用 Metal，请为以上命令加上 `--features metal`，同时运行时配合 `--device metal`。
>
> Intel MKL 预览：构建前安装 Intel oneMKL，再附加 `--features mkl`，即可在 x86 CPU 上获得更快的推理速度。
>
> 若在 Linux/Windows 上使用 NVIDIA GPU，请加上 `--features cuda` 并以 `--device cuda --dtype f16` 启动服务。

注意事项：

- 图片需使用 `data:` URL（base64）或可访问的 `http(s)` 链接，禁止本地路径。
- Server 已自动将多轮对话折叠为最近一次 user 轮次，保持单轮 OCR 体验。
- 与 [Open WebUI](https://github.com/open-webui/open-webui) 等 OpenAI 兼容客户端开箱即用——只需在客户端设置 `base_url` 为 `http://localhost:8000/v1` 并选择 `deepseek-ocr` 模型。
- 如果需要大图上传，可在 Rocket 配置里调高 JSON/body limit。

![Open WebUI 连接 deepseek-ocr.rs](./baselines/sample_1.png)

## GPU 加速 ⚡

- **Metal（macOS 13+ & Apple Silicon）**：构建命令附加 `--features metal`，运行时使用 `--device metal --dtype f16`。
- **CUDA（alpha，Linux/Windows & NVIDIA GPU）**：提前安装 CUDA 12.2+，构建时加 `--features cuda`，执行时传入 `--device cuda --dtype f16`。
- **Intel MKL（预览）**：安装 Intel oneMKL，构建时附加 `--features mkl`，可提升 x86 CPU 推理性能。
- 无论使用哪种 GPU，推荐 `cargo build --release -p deepseek-ocr-cli --features metal|cuda` 以获取更高吞吐。
- 结合 `--max-new-tokens`、`--crop-mode` 等参数可在延迟与质量之间做权衡。

## 目录结构 🗂️

- `crates/core`：推理管线、模型装载、会话模板。
- `crates/cli`：命令行入口 `deepseek-ocr-cli`。
- `crates/server`：提供 OpenAI 风格 API 的 Rocket 服务。
- `crates/assets`：模型/Tokenizer 下载与缓存工具。
- `baselines/`：基准输入输出样例，便于回归测试。

更多 CLI 说明请参见 [`crates/cli/README_CN.md`](crates/cli/README_CN.md)；服务端 API 详见 [`crates/server/README_CN.md`](crates/server/README_CN.md)。

## 常见问题 🛠️

- **资产下载源**：会自动在 Hugging Face 与 ModelScope 之间按延迟择优。命令行会提示当前使用的源与目标路径。
- **下载失败**：确认 `HF_TOKEN` 已配置，或重试以利用 Hugging Face/ModelScope 缓存。
- **首轮耗时长**：第一次推理需要加载模型并热启动 GPU（Metal/CUDA α)，后续会更快。
- **图片过大被拒**：放大 Rocket 限额或对图像进行下采样。

## 致谢 🙏

- 模型由 [DeepSeek-AI](https://huggingface.co/deepseek-ai/DeepSeek-OCR) 提供。
- 项目依赖 Candle、Rocket 等优秀 Rust 开源生态，感谢所有维护者。

## 许可证 📄

本仓库遵循上游 DeepSeek-OCR 模型的使用条款，详见 `DeepSeek-OCR/LICENSE`，下游使用请遵守相同限制。
