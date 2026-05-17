use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

/// Gather token embeddings for a batch of input ids.
///
/// `weight` must be `[vocab, hidden]` and `ids` must be rank-2 `[batch, seq]`. The returned tensor
/// has shape `[batch, seq, hidden]`.
///
/// When `weight` and `ids` reside on different devices, `ids` are temporarily moved to
/// `weight`'s device for the lookup and the result is moved back to `ids`' original device.
pub fn gather_token_embeddings(weight: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let target_device = ids.device();
    let ids = if !ids.device().same_device(weight.device()) {
        ids.to_device(weight.device())?
    } else {
        ids.clone()
    };
    ensure!(
        ids.rank() == 2,
        "input ids must have shape [batch, seq], got rank {}",
        ids.rank()
    );
    let (_vocab, hidden) = weight.shape().dims2()?;
    let (batch, seq_len) = ids.shape().dims2()?;
    let ids = if ids.dtype() == DType::I64 {
        ids
    } else {
        ids.to_dtype(DType::I64)?
    };
    let weight = weight.force_contiguous()?;
    let flat = ids.reshape((batch * seq_len,))?.force_contiguous()?;
    let gathered = weight.index_select(&flat, 0)?;
    let gathered = gathered.reshape((batch, seq_len, hidden))?;
    if !gathered.device().same_device(target_device) {
        Ok(gathered.to_device(target_device)?)
    } else {
        Ok(gathered)
    }
}
