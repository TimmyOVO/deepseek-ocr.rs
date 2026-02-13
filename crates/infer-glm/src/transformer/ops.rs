use anyhow::{Result, ensure};
use candle_core::{DType, Tensor, shape::D};
use deepseek_ocr_core::tensor::to_dtype_if_needed;

pub fn rotate_half_last_dim(tensor: &Tensor) -> Result<Tensor> {
    let mut dims = tensor.shape().dims().to_vec();
    ensure!(!dims.is_empty(), "rotate_half_last_dim expects rank >= 1");
    let last = *dims.last().expect("dims checked non-empty");
    ensure!(
        last.is_multiple_of(2),
        "rotate_half expects even dim, got {last}"
    );

    // GLM text rotary uses even/odd pairing (`x[..., 0::2]`, `x[..., 1::2]`).
    let half = last / 2;
    dims.pop();
    dims.push(half);
    dims.push(2);

    let paired = tensor.reshape(dims.as_slice())?;
    let even = paired.narrow(D::Minus1, 0, 1)?;
    let odd = paired.narrow(D::Minus1, 1, 1)?;
    let rotated = Tensor::cat(&[odd.neg()?, even], D::Minus1)?;

    let mut out_dims = tensor.shape().dims().to_vec();
    out_dims.pop();
    out_dims.push(last);
    Ok(rotated.reshape(out_dims.as_slice())?)
}

pub fn repeat_kv(hidden_states: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(hidden_states.clone());
    }
    let (batch, heads, seq_len, head_dim) = hidden_states.shape().dims4()?;
    Ok(hidden_states
        .unsqueeze(2)?
        .expand((batch, heads, repeats, seq_len, head_dim))?
        .reshape((batch, heads * repeats, seq_len, head_dim))?)
}

pub fn compute_dtype_for(tensor: &Tensor) -> DType {
    match tensor.dtype() {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    }
}

pub fn maybe_cast(tensor: &Tensor, dtype: DType) -> Result<Tensor> {
    to_dtype_if_needed(tensor, dtype)
}
