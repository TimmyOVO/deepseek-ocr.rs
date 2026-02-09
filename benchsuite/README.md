# Benchsuite

统一的基准与门禁子项目，采用“统一入口 + 模型适配器（package）”结构。

## 设计

- 统一入口：`python -m benchsuite.cli`
- 子命令：
  - `gate`：strict token 对齐门禁
  - `bench-python`：Python 侧单次基准
  - `bench-rust`：Rust CLI 侧单次基准
  - `perf`：按 model/device/precision/case matrix 自动跑 Python+Rust，对比并保存 run 历史
  - `matrix-gate`：按 model/device/precision/case matrix 执行 strict gate（prompt+token）
- 模型适配器：`benchsuite/models/<model>.py`
  - 当前实现：`glm.py`（`GlmAdapter`）

## 安装

在仓库根目录：

```bash
python -m pip install -e '.[bench]'
```

安装后可用统一命令：

```bash
benchsuite --help
```

也可以继续用模块调用：

```bash
python -m benchsuite.cli --help
```

## 离线约束

所有子命令统一设置：

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_HOME=.hf-cache`
- `TRANSFORMERS_CACHE=.hf-cache`
- `DEEPSEEK_OCR_CONFIG_DIR=.cli-config`
- `DEEPSEEK_OCR_CACHE_DIR=.cli-cache`

## 用法

### 1) strict token gate

```bash
python -m benchsuite.cli gate \
  --model glm-ocr \
  --baseline baselines/glm/matrix_v20/formula__image__n8/baseline.json \
  --rust baselines/glm/matrix_v33/formula__image__n8/rust_output.json \
  --output baselines/glm/matrix_v33/formula__image__n8/compare.json
```

### 2) 单次 Python / Rust 基准

```bash
python -m benchsuite.cli bench-python \
  --model glm-ocr \
  --model-dir .cli-cache/models/glm-ocr \
  --image baselines/sample/images/test.png \
  --prompt "Formula Recognition:" \
  --device cpu \
  --dtype f32 \
  --max-new-tokens 8 \
  --output baselines/glm/perf_py_v22/formula__test__n8/cpu_f32/bench.json

python -m benchsuite.cli bench-rust \
  --model glm-ocr \
  --cli target/release/deepseek-ocr-cli \
  --image baselines/sample/images/test.png \
  --prompt "Formula Recognition:" \
  --device cpu \
  --dtype f32 \
  --max-new-tokens 8 \
  --output baselines/glm/perf_rs_v22/formula__test__n8/cpu_f32/bench.json
```

### 3) 一键 perf 矩阵（自动跑两边 + 自动对比 + 历史 run 对比）

```bash
python -m benchsuite.cli perf \
  --run v23 \
  --include-models glm-ocr \
  --include-devices cpu mps \
  --include-precision f32 f16
```

输出包括：

- `baselines/benchsuite/runs/<run>/perf/summary.json`（结构化结果）
- `baselines/benchsuite/runs/<run>/perf/report.txt`（可读对比表）
- `baselines/benchsuite/runs/<run>/perf/<model>/<case>/<device_dtype>/{python,rust,compare}.json`

你也可以显式指定单 case：

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

快速迭代（只跑前 N 个 case）：

```bash
python -m benchsuite.cli perf \
  --run smoke \
  --include-models glm-ocr \
  --include-devices cpu \
  --include-precision f32 \
  --limit 1
```

### 4) 一键 matrix strict gate（默认 24-case）

```bash
python -m benchsuite.cli matrix-gate \
  --run gate_v34 \
  --include-models glm-ocr \
  --include-devices cpu mps \
  --include-precision f32 f16
```

输出包括：

- `baselines/benchsuite/runs/<run>/matrix/summary.json`
- `baselines/benchsuite/runs/<run>/matrix/report.txt`
- `baselines/benchsuite/runs/<run>/matrix/<model>/<case>/<device_dtype>/{python,compare}.json`

常用筛选：

```bash
python -m benchsuite.cli matrix-gate \
  --run smoke \
  --include-models glm-ocr \
  --include-devices cpu \
  --include-precision f32 \
  --limit 1

python -m benchsuite.cli matrix-gate \
  --run formula_only \
  --include-models glm-ocr \
  --include-devices cpu \
  --include-precision f32 \
  --cases formula__image__n8 formula__test__n8
```

ad-hoc 单条输入（不走内建 matrix）：

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

## 扩展新模型

1. 新建 `benchsuite/models/<name>.py`，实现 `<Name>Adapter`
2. 在 `benchsuite/registry.py` 注册名称
3. 复用统一入口，无需再新增散脚本
