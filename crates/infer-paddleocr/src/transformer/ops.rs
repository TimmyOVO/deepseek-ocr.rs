use anyhow::{Result, ensure};
use candle_core::{Tensor, shape::D};

use super::LinearWeights;

pub fn apply_linear(input: &Tensor, weights: &LinearWeights) -> Result<Tensor> {
    let (batch, seq_len, in_dim) = input.shape().dims3()?;
    let (out_dim, weight_in) = weights.weight.shape().dims2()?;
    ensure!(
        in_dim == weight_in,
        "linear weight expects input dim {} got {}",
        weight_in,
        in_dim
    );
    let flat = input.reshape((batch * seq_len, in_dim))?;
    let mut out = flat.matmul(&weights.weight.transpose(0, 1)?)?;
    if let Some(bias) = &weights.bias {
        out = out.broadcast_add(&bias.reshape((1, out_dim))?)?;
    }
    Ok(out.reshape((batch, seq_len, out_dim))?)
}

pub fn rotate_half(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    ensure!(
        !dims.is_empty(),
        "rotate_half expects tensor with rank >= 1"
    );
    let last = *dims.last().expect("dims non-empty");
    ensure!(
        last % 2 == 0,
        "rotate_half expects even hidden size, got {last}"
    );
    let half = last / 2;
    let first = tensor.narrow(D::Minus1, 0, half)?;
    let second = tensor.narrow(D::Minus1, half, half)?;
    let neg_second = second.neg()?;
    Ok(Tensor::cat(&[neg_second, first], D::Minus1)?)
}
