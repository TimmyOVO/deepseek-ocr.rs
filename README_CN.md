# deepseek-ocr.rs 🚀

Rust 实现的 DeepSeek-OCR 推理栈，提供快速 CLI 与 OpenAI 兼容的 HTTP Server，统一打包模型加载、视觉输入预处理、提示词工具与服务端能力，方便在本地 CPU、Apple Metal 或 NVIDIA CUDA GPU(alpha) 上构建文档理解工作流。

> 英文文档请参见 [README.md](README.md)。  


> 想直接下载可执行文件？访问 [Github Actions](https://github.com/TimmyOVO/deepseek-ocr.rs/actions/workflows/build-binaries.yml)，下载最新一次成功运行生成的 macOS（含 Metal）或 Windows 压缩包。

## crates/core 技术细节 🔬
- **视觉预处理**：`prepare_vision_input_from_image` 利用 `build_global_view` 构造方形全局画布，同时在启用 crop 模式时调用 `dynamic_preprocess` 进行高分辨率切片，并支持额外缩略图。
- **SAM + CLIP 融合**：`image_to_tensor` 标准化每幅图像，送入 Candle 版 SAM (`SamBackbone`) 与 CLIP-L (`ClipVisionModel`)，通过 `build_clip_sam_tokens` 将视觉特征按网格拼接保持空间对齐。
- **投影与布局 token**：自研 `ImageProjector` 对 SAM/CLIP 拼接通道做线性映射，并注入 `image_newline`/`view_separator` 学习 token，产出可直接喂入语言模型的多模态嵌入。
- **Tokenizer 对齐**：`build_prompt_tokens` 为 `<image>` 段生成与投影 token 数完全一致的占位序列（涵盖全局+局部视图），保证在裁剪多轮对话后仍与 OpenAI 风格输入兼容。
- **解码与缓存**：语言侧基于 Candle 重写 DeepSeek-V2 (`DeepseekLanguageModel`)，支持 FlashAttention、旋转位置编码与 `DynamicCache`，CLI/Server 均可高效流式输出。
- **可观测性与对齐**：调试模式暴露 CLIP/SAM trace (`VisionDebugFeatures`)，可用来与 PyTorch 官方实现逐层比对，大部分阶段已实现数值对齐；剩余的微小差异（如投影归一化、局部裁剪策略）已纳入 Roadmap，后续版本会继续收敛。

## 为什么选择 Rust？💡
官方 DeepSeek-OCR 依赖 Python + Transformers，部署体积大、依赖多，嵌入原生系统成本高。Rust 重写后的优势：
- 无需 Python 运行时或 conda，产物更小、更易嵌入。
- 内存安全、线程友好，可直接融入现有 Rust 服务。
- CLI 与 Server 共用一套核心逻辑，避免重复维护。
- 依旧兼容 OpenAI 客户端，同时聚焦单轮 OCR 场景确保输出稳定。

## 技术栈 ⚙️
- **Candle**：Rust 深度学习框架，支持 Metal/CUDA（alpha） 与 FlashAttention。
- **Rocket**：异步 HTTP 框架，提供 `/v1/responses`、`/v1/chat/completions` 等 OpenAI 兼容路由。
- **tokenizers**：上游模型提供的分词器，通过 `crates/assets` 在 Hugging Face / ModelScope 镜像间缓存与校验。
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
- **OpenAI 客户端即插即用**：Server 端自动折叠多轮对话，只保留最新 user 指令，避免 OCR 模型被多轮上下文干扰。

## 快速上手 🏁

### 环境要求
- Rust 1.78+（支持 2024 Edition）
- Git
- 可选：macOS 13+ 的 Apple Silicon（用于 Metal）
- 可选：Linux/Windows 的 NVIDIA GPU（需 CUDA 12.2+ 工具链与驱动，当前为alpha阶段）
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
不想自己编译？每次推送到 `main` 都会在 [build-binaries 工作流](https://github.com/TimmyOVO/deepseek-ocr.rs/actions/workflows/build-binaries.yml) 里产出 macOS（含 Metal）和 Windows 压缩包。登录 GitHub，打开最新一次绿色运行，下载 `deepseek-ocr-macos` 或 `deepseek-ocr-windows` 即可。

## 配置与优先级 🗂️
CLI 与 Server 共享同一份配置。首次启动会在系统配置目录生成带默认值的 `config.toml`，后续运行都会沿用该文件确保两个入口保持一致。

| 平台 | 默认配置文件 | 模型缓存目录 |
| --- | --- | --- |
| Linux | `~/.config/deepseek-ocr/config.toml` | `~/.cache/deepseek-ocr/models/<id>/…` |
| macOS | `~/Library/Application Support/deepseek-ocr/config.toml` | `~/Library/Caches/deepseek-ocr/models/<id>/…` |
| Windows | `%APPDATA%\deepseek-ocr\config.toml` | `%LOCALAPPDATA%\deepseek-ocr\models\<id>\…` |

- 可通过 `--config /path/to/config.toml`（CLI/Server 通用）自定义路径；当文件不存在时会自动创建并写入默认内容。
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
model_id = "deepseek-ocr"
```

- `[models]` 用于指定当前激活的模型以及额外的模型条目（每个条目都可以指向各自的配置、分词器与权重文件）。
- `[inference]` 提供 CLI 与 Server 共用的推理默认值（设备、模板、视觉分辨率、生成长度与缓存策略）。
- `[server]` 决定网络监听地址以及 `/v1/models` 返回的模型名。

更多覆盖项详见 `crates/cli/README_CN.md` 与 `crates/server/README_CN.md`。

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

## HTTP Server ☁️
启动 OpenAI 兼容服务：
```bash
cargo run -p deepseek-ocr-server --release -- \
  --host 0.0.0.0 --port 8000 \
  --device cpu --max-new-tokens 512
```
> 如果要在 macOS 上启用 Metal，请为以上命令加上 `--features metal`，同时运行时配合 `--device metal`。
>
> 若在 Linux/Windows 上使用 NVIDIA GPU，请加上 `--features cuda` 并以 `--device cuda --dtype f16` 启动服务。

注意事项：
- 图片需使用 `data:` URL（base64）或可访问的 `http(s)` 链接，禁止本地路径。
- Server 已自动将多轮对话折叠为最近一次 user 轮次，保持单轮 OCR 体验。
- 与 [Open WebUI](https://github.com/open-webui/open-webui) 等 OpenAI 兼容客户端开箱即用——只需在客户端设置 `base_url` 为 `http://localhost:8000/v1` 并选择 `deepseek-ocr` 模型。
- 如果需要大图上传，可在 Rocket 配置里调高 JSON/body limit。

![Open WebUI 连接 deepseek-ocr.rs](./assets/sample_1.png)

## GPU 加速 ⚡
- **Metal（macOS 13+ & Apple Silicon）**：构建命令附加 `--features metal`，运行时使用 `--device metal --dtype f16`。
- **CUDA（alpha，Linux/Windows & NVIDIA GPU）**：提前安装 CUDA 12.2+，构建时加 `--features cuda`，执行时传入 `--device cuda --dtype f16`。
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

## Roadmap 🗺️
- ✅ Apple Metal 后端 + FP16 支持，CLI/Server 已在 macOS 上对齐。
- ✅ NVIDIA CUDA 后端(alpha)：`--features cuda` + `--device cuda --dtype f16` 可在 Linux/Windows 上尝鲜 GPU 加速。
- 🔄 **对齐完善**：完成投影归一化、局部裁剪等细节的数值校准，并扩展中间张量对比用例。
- 🔄 **Grounding 与流式体验**：移植 Python 版的框选/Markdown 后处理，提升 SSE 流式交互体验。
- 🔄 **跨平台加速**：继续调优 CUDA 性能、补齐 CPU/Metal/CUDA 自动检测，并发布可选 GPU 基准测试。
- 🔄 **打包与运维**：提供带校验的二进制发行版，增强日志/指标，并补充 Helm/Docker 部署示例。
- 🔜 **结构化输出**：在对齐完成后引入可选 JSON Schema 工具，方便下游自动化。

## 致谢 🙏
- 模型由 [DeepSeek-AI](https://huggingface.co/deepseek-ai/DeepSeek-OCR) 提供。
- 项目依赖 Candle、Rocket 等优秀 Rust 开源生态，感谢所有维护者。

## 许可证 📄
本仓库遵循上游 DeepSeek-OCR 模型的使用条款，详见 `DeepSeek-OCR/LICENSE`，下游使用请遵守相同限制。
