# Benchsuite

`benchsuite` 是仓库内统一的 **性能对比 + 严格门禁** 子项目。

## 安装与入口

在仓库根目录执行：

```bash
python -m pip install -e '.[bench]'
```

统一入口：

```bash
python -m benchsuite.cli --help
```

## 支持模型（仅 Rust infer 已支持的 baseline）

当前 benchsuite 默认接入与 Rust infer 一致的 5 个 baseline 模型：

- `deepseek-ocr`
- `deepseek-ocr-2`
- `paddleocr-vl`
- `dots-ocr`
- `glm-ocr`

> 量化模型（`*-q4k/q6k/q8k`）不参与 benchsuite 对比：
> 量化不是同精度路径，无法做 strict 一致性验证，且没有对等 Python baseline。

## 能力矩阵（python / rust / strict）

| model_id | python baseline | rust bench/infer | strict compare |
| --- | --- | --- | --- |
| `glm-ocr` | ✅ | ✅ | ✅ |
| `deepseek-ocr` | ❌ | ✅ | ❌ |
| `deepseek-ocr-2` | ❌ | ✅ | ❌ |
| `paddleocr-vl` | ❌ | ✅ | ❌ |
| `dots-ocr` | ❌ | ✅ | ❌ |

说明：

- strict compare 只在“同设备同精度 + 有对等 Python baseline”时执行。
- 无法 strict 时会在 `summary.json` / `report.txt` 中结构化记录 `strict_status=skipped` 与 `skip_reason`。

## 统一运行时目录（不硬编码仓库隐藏目录）

benchsuite 不再写死 `.hf-cache/.cli-cache/.cli-config`。

它会使用统一 runtime root（默认：`/tmp/deepseek-ocr-benchsuite`）并在其下创建：

- `huggingface/`（`HF_HOME` / `TRANSFORMERS_CACHE` / `HUGGINGFACE_HUB_CACHE`）
- `deepseek-ocr-config/`（`DEEPSEEK_OCR_CONFIG_DIR`）
- `deepseek-ocr-cache/`（`DEEPSEEK_OCR_CACHE_DIR`）

可通过以下方式覆盖：

- CLI 参数：`--runtime-root /path/to/runtime`
- 环境变量：`BENCHSUITE_RUNTIME_ROOT` 或 `DEEPSEEK_OCR_RUNTIME_ROOT`

## 矩阵语义（同设备同精度）

`perf` / `matrix-gate` 继续支持：

- `--include-models`
- `--include-devices`（`cpu` / `mps`）
- `--include-precision`（`f32` / `f16`）

默认行为（不传时）：

- 模型：5 个 baseline 全量
- 设备：`cpu mps`
- 精度：`f32 f16`

过滤规则：

- `mps` 不可用时自动跳过；
- `cpu + f16` 默认跳过；
- Python 设备 `mps` 会映射到 Rust 设备 `metal`。

## 子命令

### `perf`

自动运行（按能力自动决定 compare/skip）：

```bash
python -m benchsuite.cli perf \
  --run smoke_models_perf_v1 \
  --include-models glm-ocr \
  --include-devices cpu \
  --include-precision f32 \
  --limit 1
```

多模型：

```bash
python -m benchsuite.cli perf \
  --run smoke_models_perf_v1 \
  --include-models deepseek-ocr deepseek-ocr-2 paddleocr-vl dots-ocr glm-ocr \
  --include-devices cpu \
  --include-precision f32 \
  --limit 1
```

指定 runtime root：

```bash
python -m benchsuite.cli perf \
  --run smoke_models_perf_v1 \
  --runtime-root /tmp/deepseek-ocr-bench-runtime \
  --include-models glm-ocr \
  --include-devices cpu \
  --include-precision f32 \
  --limit 1
```

### `matrix-gate`

严格门禁（仅可比模型会执行 strict）：

```bash
python -m benchsuite.cli matrix-gate \
  --run smoke_models_gate_v1 \
  --include-models deepseek-ocr deepseek-ocr-2 paddleocr-vl dots-ocr glm-ocr \
  --include-devices cpu \
  --include-precision f32 \
  --limit 1
```

## 常见失败与排查

- `strict_status=skipped`
  - 这是能力分层行为，不是 silent skip。
  - 查看 `skip_reason`：通常是无 Python baseline 或无同精度可比路径。

- `python baseline import/config error`
  - 多见于模型在 Transformers 端没有可用等价 baseline 或依赖不满足。
  - 此时 `perf` 仍可输出 Rust 指标，strict 会被结构化跳过。

- `rust cli 下载/加载失败`
  - 确认 runtime root 可写；
  - 确认网络可访问模型源（若你主动设置了 offline 变量则需本地已有缓存）。

- `mps/f16 对不齐`
  - benchsuite 仅比较同设备同精度；无可运行 pair 会直接报错，避免错误比较。
