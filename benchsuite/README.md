# Benchsuite

`benchsuite` 是仓库内统一的 **性能对比 + 严格门禁** 子项目。

## 安装与入口


```bash
python -m pip install -e '.[bench]'
```

统一入口：

```bash
python -m benchsuite.cli --help
```

## 支持模型（与 Rust infer 对齐）

当前默认模型：

- `deepseek-ocr`
- `deepseek-ocr-2`
- `paddleocr-vl`
- `dots-ocr`
- `glm-ocr`

## 能力矩阵（python / rust / strict）

| model_id | python baseline | rust bench/infer | strict compare |
| --- | --- | --- | --- |
| `glm-ocr` | ✅ | ✅ | ✅ |
| `deepseek-ocr` | ✅ | ✅ | ✅ |
| `deepseek-ocr-2` | ✅ | ✅ | ✅ |
| `paddleocr-vl` | ❌ | ✅ | ❌ |
| `dots-ocr` | ❌ | ✅ | ❌ |

说明：

- strict compare 只在“同设备同精度 + Python/Rust 两侧都有结果”时执行。
- 不可比较时会在 `summary.json` / `report.txt` 记录 `strict_status=skipped` 与 `skip_reason`。

## 多模型依赖隔离（自动创建 Python 子环境）

不同模型的 Python baseline 依赖可能互相冲突。`benchsuite` 现在默认走统一 runtime root 下的自动子环境：

- `runtime_root/python-envs/glm`（用于 `glm-ocr`）
- `runtime_root/python-envs/deepseek`（用于 `deepseek-ocr` / `deepseek-ocr-2`）

可通过以下方式覆盖：

- CLI 参数：`--runtime-root /path/to/runtime`
- 环境变量：`BENCHSUITE_RUNTIME_ROOT` 或 `DEEPSEEK_OCR_RUNTIME_ROOT`

## 矩阵语义（同设备同精度）

`perf` / `matrix-gate` 支持：

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
- Python 设备 `mps` 映射到 Rust 设备 `metal`。

## 子命令

### `perf`

自动运行 Rust + Python（按能力比较/跳过）并输出控制台表格：

```bash
python -m benchsuite.cli perf \
  --run smoke_models_perf_v1 \
  --include-models deepseek-ocr deepseek-ocr-2 paddleocr-vl dots-ocr glm-ocr \
  --include-devices cpu \
  --include-precision f32 \
  --limit 1
```

### `matrix-gate`

严格门禁（只对可比 pair 执行 strict）：

```bash
python -m benchsuite.cli matrix-gate \
  --run smoke_models_gate_v1 \
  --include-models deepseek-ocr deepseek-ocr-2 paddleocr-vl dots-ocr glm-ocr \
  --include-devices cpu \
  --include-precision f32 \
  --limit 1
```
