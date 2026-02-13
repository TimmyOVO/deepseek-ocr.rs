use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};

/// Returns `tensor` cast to `dtype` only when needed.
pub fn to_dtype_if_needed(tensor: &Tensor, dtype: DType) -> Result<Tensor> {
    if tensor.dtype() == dtype {
        Ok(tensor.clone())
    } else {
        Ok(tensor.to_dtype(dtype)?)
    }
}

/// Returns owned `tensor` cast to `dtype` only when needed.
pub fn into_dtype_if_needed(tensor: Tensor, dtype: DType) -> Result<Tensor> {
    if tensor.dtype() == dtype {
        Ok(tensor)
    } else {
        Ok(tensor.to_dtype(dtype)?)
    }
}

/// Casts each tensor to `dtype` only when needed.
pub fn into_dtype_vec_if_needed(tensors: Vec<Tensor>, dtype: DType) -> Result<Vec<Tensor>> {
    tensors
        .into_iter()
        .map(|tensor| into_dtype_if_needed(tensor, dtype))
        .collect()
}

/// Returns `tensor` moved to `device` only when needed.
pub fn to_device_if_needed(tensor: &Tensor, device: &Device) -> Result<Tensor> {
    if tensor.device().same_device(device) {
        Ok(tensor.clone())
    } else {
        Ok(tensor.to_device(device)?)
    }
}

/// Returns owned `tensor` moved to `device` only when needed.
pub fn into_device_if_needed(tensor: Tensor, device: &Device) -> Result<Tensor> {
    if tensor.device().same_device(device) {
        Ok(tensor)
    } else {
        Ok(tensor.to_device(device)?)
    }
}

/// Gather token embeddings for a batch of input ids.
///
/// `weight` must be `[vocab, hidden]` and `ids` must be rank-2 `[batch, seq]`. The returned tensor
/// has shape `[batch, seq, hidden]`.
pub fn gather_token_embeddings(weight: &Tensor, ids: &Tensor) -> Result<Tensor> {
    ensure!(
        ids.rank() == 2,
        "input ids must have shape [batch, seq], got rank {}",
        ids.rank()
    );
    let (_vocab, hidden) = weight.shape().dims2()?;
    let (batch, seq_len) = ids.shape().dims2()?;
    let ids = to_dtype_if_needed(ids, DType::I64)?;
    let weight = weight.force_contiguous()?;
    let flat = ids.reshape((batch * seq_len,))?.force_contiguous()?;
    let gathered = weight.index_select(&flat, 0)?;
    Ok(gathered.reshape((batch, seq_len, hidden))?)
}
