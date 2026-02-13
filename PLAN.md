
以下是重构的**High-Level Architecture**设计，分为四个阶段。每个阶段都包含**核心目标**、**Rust 最佳实践代码示例**以及**验收标准（Definition of Done）**。

---

### High-Level Architecture: `deepseek-ocr-layers`

我们将创建一个核心库 `deepseek-ocr-layers`，它位于 `candle-core` 之上，`infer-*` 具体业务逻辑之下。

```mermaid
graph TD
    A[Application Layer: infer-deepseek / infer-dots] --> B[Model Assembly Layer]
    B --> C[Standard Blocks: DecoderLayer / VisionBlock]
    C --> D[Functional Components: Attention / MLP / RoPE]
    D --> E[Primitives: Linear / RMSNorm / FusedOps]
    E --> F[Memory Management: StaticKVCache / Allocator]
    F --> G[Candle Core / Custom Kernels (CUDA/Metal)]

```

---

### 阶段一：基础类型与统一配置 (Foundation & Configuration)

**目标：** 消除不同 crate 中重复定义的 Config 结构，建立统一的模型描述语言，解耦 JSON 解析与内部配置。

#### 1. 核心设计

使用 **Canonical Config Pattern**。不要直接在推理逻辑中使用 serde 解析出来的 struct。解析层（DTO）和 运行时配置（Runtime Config）分离。

#### 2. Rust 最佳实践代码示例

```rust
// crate: deepseek-ocr-layers/src/config.rs

use serde::Deserialize;
use thiserror::Error;

/// 定义架构类型，决定了由哪个 Pipeline 处理
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    DeepSeekV2,
    Qwen2,
    Llama3,
    ViT,
}

/// 运行时配置 (Runtime Config) - 这是我们代码中传递的核心对象
/// 使用 Builder 模式构造，保证构建时的校验
#[derive(Debug, Clone)]
pub struct OcrModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rope_config: RopeConfig,
    pub eps: f64,
}

#[derive(Debug, Clone)]
pub enum RopeConfig {
    Standard { theta: f32 },
    MultiModal { sections: Vec<usize>, theta: f32 }, // 适配 PaddleOCR
}

// ----------------------------------------------------------------
// DTO (Data Transfer Objects) - 仅用于从 JSON 反序列化，容忍脏数据
// ----------------------------------------------------------------
#[derive(Deserialize)]
struct RawDeepseekConfig {
    n_routed_experts: Option<usize>,
    // Deepseek 叫 num_attention_heads
    num_attention_heads: usize, 
    // ...
}

#[derive(Deserialize)]
struct RawQwenConfig {
    // Qwen 可能叫 n_heads
    n_heads: Option<usize>, 
    // ...
}

// 通过 TryFrom 实现防腐层 (Anti-Corruption Layer)
impl TryFrom<RawDeepseekConfig> for OcrModelConfig {
    type Error = anyhow::Error;
    fn try_from(raw: RawDeepseekConfig) -> Result<Self, Self::Error> {
        // 在这里处理所有默认值逻辑、校验逻辑
        Ok(Self {
            hidden_size: raw.hidden_size,
            // ...
            rope_config: RopeConfig::Standard { theta: 10000.0 },
        })
    }
}

```

#### 3. 收尾标准 (Definition of Done)

1. [ ] 所有 `infer-*` crate 不再直接依赖 `serde_json` 解析出来的结构体进行推理，而是依赖 `OcrModelConfig`。
2. [ ] 所有关于 `Option<usize>` 的 `unwrap_or` 逻辑全部移动到 `TryFrom` 转换层，Runtime Config 中不应包含非必要的 Option。
3. [ ] 单元测试覆盖不同 JSON 格式到 `OcrModelConfig` 的转换正确性。

---

### 阶段二：内存管理重构 (Memory & RSS Optimization)

**目标：** 解决 RSS 高和 Decode 慢的核心问题——**KV Cache 碎片化**。从动态 `Vec` 转向静态预分配。

#### 1. 核心设计

实现一个 **Static KV Cache**。在模型加载时，根据用户设定的 `max_batch_size` 和 `max_seq_len` 一次性申请显存。使用 `Ring Buffer` 或 `Rolling Index` 管理写入位置。

#### 2. Rust 最佳实践代码示例

使用 **Interior Mutability (内部可变性)** 和 **Generic Array** 思想。

```rust
// crate: deepseek-ocr-layers/src/cache.rs

use candle_core::{Device, DType, Tensor, Result};

/// 定义 Cache 的存储后端 trait，方便未来扩展到 PagedAttention
pub trait KVCacheStorage: Send + Sync {
    fn append(&self, layer_idx: usize, k: &Tensor, v: &Tensor, pos: usize) -> Result<()>;
    fn get_view(&self, layer_idx: usize, current_len: usize) -> Result<(Tensor, Tensor)>;
    fn reset(&self);
}

pub struct StaticKVCache {
    k_buffer: Tensor, // Shape: [n_layers, batch, max_seq, heads, head_dim]
    v_buffer: Tensor,
    max_seq_len: usize,
    // 使用 Atomic 或 Mutex 管理当前长度，避免 &mut self 穿透整个调用栈
    current_pos: std::sync::atomic::AtomicUsize, 
}

impl StaticKVCache {
    pub fn new(cfg: &OcrModelConfig, max_seq_len: usize, batch_size: usize, device: &Device, dtype: DType) -> Result<Self> {
        // 关键优化：在这里一次性分配大块显存，减少 malloc 碎片
        let k_shape = (
            cfg.num_hidden_layers,
            batch_size,
            cfg.num_kv_heads,
            max_seq_len,
            cfg.head_dim
        );
        // ... v_shape 类似
        
        let k_buffer = Tensor::zeros(k_shape, dtype, device)?;
        let v_buffer = Tensor::zeros(v_shape, dtype, device)?;
        
        Ok(Self {
            k_buffer,
            v_buffer,
            max_seq_len,
            current_pos: 0.into(),
        })
    }
}

impl KVCacheStorage for StaticKVCache {
    fn append(&self, layer_idx: usize, k: &Tensor, v: &Tensor, start_pos: usize) -> Result<()> {
        // 使用 slice_assign 将新 token 的 KV 写入预分配的 buffer
        // 这是一个 Zero-Copy 操作（相对于重新申请内存）
        let seq_len = k.dim(1)?;
        // bounds check...
        
        // 伪代码：self.k_buffer.i(layer_idx).narrow(2, start_pos, seq_len).assign(k)
        Ok(())
    }
    
    // ...
}

```

#### 3. 收尾标准 (Definition of Done)

1. [ ] 实现 `StaticKVCache`，并在 `infer-deepseek` 中替换掉原来的 `DynamicCache`。
2. [ ] **性能指标：** Decode 阶段 RSS 波动曲线变平（不再随 token 增长而线性增长，而是一开始就固定）。
3. [ ] **性能指标：** 在长序列（>2048 token）Decode 时，显存分配耗时减少 90% 以上。

---

### 阶段三：算子抽象与融合 (Primitives & Fusion)

**目标：** 解决 Decode 带宽瓶颈，移除 Rust 层面的 F32 Cast。

#### 1. 核心设计

定义 **Op Traits**，并提供 `Fused` 实现。利用 Rust 的 **Generic Constraints** 来保证类型安全。

#### 2. Rust 最佳实践代码示例

```rust
// crate: deepseek-ocr-layers/src/ops.rs

use candle_core::{Tensor, Result};

/// 归一化层接口：屏蔽 RMSNorm / LayerNorm 以及 eps 的差异
pub trait Normalization: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

/// 针对 RMSNorm 的 Fused 实现
pub struct FusedRmsNorm {
    scale: Tensor,
    eps: f64,
}

impl FusedRmsNorm {
    pub fn new(scale: Tensor, eps: f64) -> Self { Self { scale, eps } }
}

impl Normalization for FusedRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 关键优化：调用自定义的 CUDA/Metal Kernel
        // 该 Kernel 接收 F16 输入，内部用 F32 累加，输出 F16
        // 避免了 x.to_f32() -> calc -> to_f16() 的带宽浪费
        #[cfg(feature = "cuda")]
        return candle_nn::ops::rms_norm_fused(x, &self.scale, self.eps);
        
        #[cfg(not(feature = "cuda"))]
        return candle_nn::ops::rms_norm(x, &self.scale, self.eps);
    }
}

/// 线性层接口：屏蔽 Quantized / Float 的差异
pub trait LinearLayer: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

pub enum DynLinear {
    Float(candle_nn::Linear),
    Quantized(candle_core::quantized::QMatMul),
}

impl LinearLayer for DynLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            // 自动处理 F16 -> F32 cast 逻辑，如果底层 kernel 不支持
            Self::Float(l) => l.forward(x),
            Self::Quantized(q) => {
                // 在这里集中处理量化矩阵乘法的 dequant 逻辑
                // 确保只在必要时 cast
                crate::quant::matmul_fused(x, q)
            }
        }
    }
}

```

#### 3. 融合算子建议 (Fused Kernels)

你需要重点实现（或调用 `candle-nn` 现有的）以下融合算子：

1. **`ApplyRotaryEmb`**: 不要生成 cos/sin table 再 broadcast mul。直接写一个 kernel，读取 Q, K 和 Position ID，原地计算并旋转。这对 Decode 速度提升巨大。
2. **`SwiGLU`**: `Silu(x * w1) * (x * w2)`。这涉及三次 kernel launch。融合为一个 kernel 可以减少 2/3 的显存读写。
3. **`RMSNorm`**: 必须支持 Mixed Precision (F16 in, F32 accum, F16 out)。

#### 4. 收尾标准 (Definition of Done)

1. [ ] 代码中不再出现 `x.to_dtype(DType::F32)` 用于计算 Norm 或 Residual 的代码。
2. [ ] 实现统一的 `DynLinear` 封装，替换掉原来散落在各处的 `SnapshotLinear` 逻辑。
3. [ ] **性能指标：** Decode 阶段 GPU Kernel 占用率（Compute Utilization）提升，Memory Bandwidth 利用率接近硬件上限。

---

### 阶段四：组件组装与统一推理 (Assembly)

**目标：** 消除 Copy-Paste 的 Transformer Block 代码。

#### 1. 核心设计

使用 **Composition over Inheritance**。`DeepseekModel` 只是一个配置好的 Block 容器。

#### 2. Rust 最佳实践代码示例

```rust
// crate: deepseek-ocr-layers/src/model.rs

pub struct DecoderLayer {
    self_attn: Box<dyn AttentionModule>,
    mlp: Box<dyn MLPModule>,
    input_norm: Box<dyn Normalization>,
    post_attn_norm: Box<dyn Normalization>,
}

impl DecoderLayer {
    pub fn forward(&self, x: &Tensor, cache: &KVCacheStorage, pos: &Tensor) -> Result<Tensor> {
        // 经典的 Transformer 结构，逻辑复用
        let residual = x;
        let x_norm = self.input_norm.forward(x)?;
        
        // Attention 内部封装了 RoPE 和 FlashAttn/Naive 的选择
        let attn_out = self.self_attn.forward(&x_norm, cache, pos)?;
        
        let x = (residual + attn_out)?;
        
        let residual = &x;
        let x_norm = self.post_attn_norm.forward(&x)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        
        let x = (residual + mlp_out)?;
        Ok(x)
    }
}

// 模型定义变得非常简洁
pub struct GenericOcrModel {
    layers: Vec<DecoderLayer>,
    embed: Box<dyn Embedding>,
    head: Box<dyn LinearLayer>,
    norm: Box<dyn Normalization>,
}

```

#### 3. 收尾标准 (Definition of Done)

1. [ ] `infer-deepseek`, `infer-paddle`, `infer-glm` 删除了自己的 `transformer` 模块，全部引用 `deepseek-ocr-layers`。
2. [ ] 代码行数减少 40% 以上。
3. [ ] 新增一个模型架构（如 Llama3）只需要编写 Config 转换和权重加载逻辑，不需要写推理逻辑。

---

### 总结建议：执行顺序

1. **Stop the bleeding:** 先做 **Phase 3 (Primitives)** 中的 **Removal of F32 Casts**。这是性价比最高的，不需要大规模重构架构就能看到速度提升。
2. **Memory Fix:** 接着做 **Phase 2 (Static Cache)**。这将立即解决 RSS 问题。
3. **Refactor:** 最后做 **Phase 1 & 4**。这是为了长期的可维护性。

**关于融合算子（Fused Ops）：**
建议在 `crates/kernels/` 下使用 `bindgen` + `cc` crate 编译自定义的 CUDA (`.cu`) 和 Metal (`.metal`) 代码，并通过 FFI 暴露给 Rust。不要试图用纯 Rust 模拟复杂的融合逻辑，那无法达到极致性能。

你现在可以按照这个 roadmap，从 **Remove F32 Casts** 开始着手了。