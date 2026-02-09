# Benchsuite

`benchsuite` 是仓库内统一的 **性能对比 + 严格门禁** 子项目

---

## 安装与运行入口

在仓库根目录执行：

```bash
python -m pip install -e '.[bench]'
```

安装后可用两种入口：

```bash
benchsuite --help
python -m benchsuite.cli --help
```

---

## 矩阵语义（同设备同精度）

`perf` 和 `matrix-gate` 都支持：

- `--include-models`（数组）
- `--include-devices`（数组，当前支持 `cpu`/`mps`）
- `--include-precision`（数组，当前支持 `f32`/`f16`）

默认行为（不传时）：

- 模型：来自 `registry.list_default_models()`（当前为 `glm-ocr`）
- 设备：`cpu mps`
- 精度：`f32 f16`

当前过滤规则（`BaseAdapter`）：

- `mps` 不可用时自动跳过；
- `cpu + f16` 默认禁用；
- Python 设备 `mps` 会映射到 Rust 设备 `metal`。

---

## 子命令说明

### `gate`

用途：对比 baseline 与 Rust 输出的 token 严格一致性。

```bash
python -m benchsuite.cli gate \
  --model glm-ocr \
  --baseline baselines/glm/matrix_v20/formula__image__n8/baseline.json \
  --rust baselines/glm/matrix_v33/formula__image__n8/rust_output.json \
  --output baselines/glm/matrix_v33/formula__image__n8/compare.json
```

说明：

- `gate` 走 adapter 的 `compare_tokens`，聚焦 token 严格一致。
- 退出码：`match=true` 返回 0，否则返回 1。

### `bench-python`

用途：单条 case 跑 Python baseline 并输出结构化指标。

```bash
python -m benchsuite.cli bench-python \
  --model glm-ocr \
  --model-dir .cli-cache/models/glm-ocr \
  --image baselines/sample/images/test.png \
  --prompt "Formula Recognition:" \
  --device cpu \
  --dtype f32 \
  --max-new-tokens 8 \
  --output /tmp/py_bench.json
```

### `bench-rust`

用途：单条 case 跑 Rust CLI benchmark 并输出结构化指标。

```bash
python -m benchsuite.cli bench-rust \
  --model glm-ocr \
  --cli target/release/deepseek-ocr-cli \
  --image baselines/sample/images/test.png \
  --prompt "Formula Recognition:" \
  --device cpu \
  --dtype f32 \
  --max-new-tokens 8 \
  --output /tmp/rs_bench.json
```

### `perf`

用途：自动跑 Python + Rust，做 strict 对比并输出性能表。

最小示例：

```bash
python -m benchsuite.cli perf --run v1
```

常用筛选：

```bash
python -m benchsuite.cli perf \
  --run smoke \
  --include-models glm-ocr \
  --include-devices cpu \
  --include-precision f32 \
  --limit 1
```

ad-hoc 输入（不走默认 case matrix）：

```bash
python -m benchsuite.cli perf \
  --run adhoc \
  --include-models glm-ocr \
  --include-devices cpu \
  --include-precision f32 \
  --image baselines/sample/images/test.png \
  --prompt "Formula Recognition:" \
  --max-new-tokens 64
```

case 来源优先级：

1. `--image + --prompt`（ad-hoc）
2. `--baseline-json` 或 `--case-name`
3. adapter 默认 case matrix（支持 `--cases` / `--limit`）

### `matrix-gate`

用途：按矩阵批量执行严格门禁（prompt + token）。

最小示例：

```bash
python -m benchsuite.cli matrix-gate --run gate_v1
```

快速迭代：

```bash
python -m benchsuite.cli matrix-gate \
  --run gate_smoke \
  --include-models glm-ocr \
  --include-devices cpu \
  --include-precision f32 \
  --limit 1
```

ad-hoc 输入：

```bash
python -m benchsuite.cli matrix-gate \
  --run adhoc_gate \
  --include-models glm-ocr \
  --include-devices cpu \
  --include-precision f32 \
  --image baselines/sample/images/test.png \
  --prompt "Formula Recognition:" \
  --max-new-tokens 8
```


