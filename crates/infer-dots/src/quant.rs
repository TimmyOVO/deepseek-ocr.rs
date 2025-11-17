use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Module, Tensor, quantized::QMatMul};
use candle_nn::VarBuilder;
use tracing::trace;

use crate::snapshot::{SnapshotLinear, SnapshotLinearMap};

/// Fully-connected layer that can be backed either by a dense weight matrix
/// or a quantized `QMatMul` reconstructed from a DSQ snapshot.
///
/// This mirrors the lightweight `LinearWeights` wrappers used in the
/// DeepSeek/Paddle backends but is scoped to the DotsOCR implementation.
#[derive(Debug, Clone)]
pub struct QuantLinear {
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub qmatmul: Option<Arc<QMatMul>>,
    pub out_dim: usize,
    pub in_dim: usize,
    #[allow(dead_code)]
    pub label: String,
}

impl QuantLinear {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: VarBuilder,
        out_dim: usize,
        in_dim: usize,
        use_bias: bool,
        snapshot_hits: Option<&mut SnapshotLinearMap>,
        snapshot_label: Option<&'static str>,
    ) -> Result<Self> {
        let label = qualified_name(&vb, "weight");
        let device = vb.device();
        let mut weight: Option<Tensor> = None;
        let mut bias: Option<Tensor> = None;
        let mut qmatmul: Option<Arc<QMatMul>> = None;

        if let Some(hit) = snapshot_hits.and_then(|hits| hits.remove(&label)) {
            match hit {
                SnapshotLinear::Quantized {
                    qmatmul: qm,
                    bias: snap_bias,
                } => {
                    trace!(
                        tensor = label,
                        backend = backend_label(device),
                        container = snapshot_label.unwrap_or("snapshot"),
                        source = "snapshot",
                        action = "quantized",
                        "dots-linear"
                    );
                    qmatmul = Some(qm);
                    bias = snap_bias;
                }
                SnapshotLinear::Float {
                    weight: snap_weight,
                    bias: snap_bias,
                } => {
                    trace!(
                        tensor = label,
                        backend = backend_label(device),
                        container = snapshot_label.unwrap_or("snapshot"),
                        source = "snapshot",
                        action = "float",
                        "dots-linear"
                    );
                    weight = Some(snap_weight);
                    bias = snap_bias;
                }
            }
        }

        if weight.is_none() && qmatmul.is_none() {
            weight = Some(
                vb.get((out_dim, in_dim), "weight")
                    .with_context(|| format!("missing linear weight `{label}`"))?
                    .contiguous()?,
            );
        }

        if bias.is_none() && use_bias && vb.contains_tensor("bias") {
            bias = Some(
                vb.get(out_dim, "bias")
                    .with_context(|| {
                        format!("missing linear bias `{}`", qualified_name(&vb, "bias"))
                    })?
                    .contiguous()?,
            );
        }

        Ok(Self {
            weight,
            bias,
            qmatmul,
            out_dim,
            in_dim,
            label,
        })
    }

    /// Apply the linear layer to a 2D activation matrix `[rows, in_dim]`.
    pub fn matmul_2d(&self, input: &Tensor) -> Result<Tensor> {
        if let Some(qm) = &self.qmatmul {
            run_quantized_matmul(qm, input)
        } else {
            let weight = self
                .weight
                .as_ref()
                .context("linear weight missing for float matmul")?;
            let mut transposed = weight.transpose(0, 1)?;
            if transposed.dtype() != input.dtype() {
                transposed = transposed.to_dtype(input.dtype())?;
            }
            Ok(input.matmul(&transposed)?)
        }
    }

    /// Apply the linear layer to an arbitrary-rank tensor whose last
    /// dimension matches `in_dim`, preserving the leading dimensions.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims().to_vec();
        ensure!(!dims.is_empty(), "dots linear expects rank >= 1");
        let last = *dims.last().expect("non-empty dims");
        ensure!(
            last == self.in_dim,
            "dots linear expected last dim {} got {}",
            self.in_dim,
            last
        );

        let outer: usize = if dims.len() == 1 {
            1
        } else {
            dims[..dims.len() - 1].iter().product()
        };
        let reshaped = input.reshape((outer, self.in_dim))?;
        let mut out = self.matmul_2d(&reshaped)?;

        if let Some(bias) = &self.bias {
            let bias = if bias.dtype() == out.dtype() {
                bias.clone()
            } else {
                bias.to_dtype(out.dtype())?
            };
            out = out.broadcast_add(&bias.reshape((1, self.out_dim))?)?;
        }

        let mut new_dims = dims;
        let last_idx = new_dims.len() - 1;
        new_dims[last_idx] = self.out_dim;
        let mut restored = out.reshape(new_dims)?;
        if restored.dtype() != input.dtype() {
            restored = restored.to_dtype(input.dtype())?;
        }
        Ok(restored)
    }
}

/// Simple backend label helper used in trace logs.
fn backend_label(device: &candle_core::Device) -> &'static str {
    if device.is_cuda() {
        "CUDA"
    } else if device.is_metal() {
        "Metal"
    } else {
        "CPU"
    }
}

/// Quantized matmul helper mirroring the behaviour of the DeepSeek backend:
/// - on Metal/CUDA, activations are upcast to `F32` before running the kernel;
/// - output is cast back to the original input dtype.
pub fn run_quantized_matmul(qm: &QMatMul, input: &Tensor) -> Result<Tensor> {
    let dtype = input.dtype();
    let device = input.device();
    if device.is_cuda() || device.is_metal() {
        let mut out = if dtype == DType::F32 {
            qm.forward(input)?
        } else {
            let activations = input.to_dtype(DType::F32)?;
            qm.forward(&activations)?
        };
        if out.dtype() != dtype {
            out = out.to_dtype(dtype)?;
        }
        Ok(out)
    } else {
        let mut out = qm.forward(input)?;
        if out.dtype() != dtype {
            out = out.to_dtype(dtype)?;
        }
        Ok(out)
    }
}

/// Compose a fully-qualified tensor name from a [`VarBuilder`] prefix and
/// the tensor suffix. This must match how safetensors names are constructed.
pub fn qualified_name(vb: &VarBuilder, tensor: &str) -> String {
    let prefix = vb.prefix();
    if prefix.is_empty() {
        tensor.to_string()
    } else {
        format!("{prefix}.{tensor}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn quant_linear_forward_preserves_last_dim() -> Result<()> {
        let device = Device::Cpu;
        let out_dim = 4;
        let in_dim = 3;
        let weight = Tensor::zeros((out_dim, in_dim), DType::F32, &device)?;
        let bias = Tensor::zeros(out_dim, DType::F32, &device)?;
        let linear = QuantLinear {
            weight: Some(weight),
            bias: Some(bias),
            qmatmul: None,
            out_dim,
            in_dim,
            label: "test.linear.weight".to_string(),
        };
        let input = Tensor::zeros((2, 5, in_dim), DType::F32, &device)?;
        let out = linear.forward(&input)?;
        assert_eq!(out.shape().dims(), &[2, 5, out_dim]);
        Ok(())
    }
}
