use anyhow::{Result, ensure};
use candle_core::{Tensor, shape::D};

use super::LinearWeights;

pub fn apply_linear(input: &Tensor, weights: &LinearWeights) -> Result<Tensor> {
    let (batch, seq_len, in_dim) = input.shape().dims3()?;
    ensure!(
        in_dim == weights.in_dim,
        "linear weight expects input dim {} got {}",
        weights.in_dim,
        in_dim
    );
    let flat = input.reshape((batch * seq_len, in_dim))?;
    let mut out = weights.matmul_2d(&flat)?;
    if let Some(bias) = &weights.bias {
        out = out.broadcast_add(&bias.reshape((1, weights.out_dim))?)?;
    }
    let out_dim = weights.out_dim;
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
