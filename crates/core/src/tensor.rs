use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

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
    let ids = if ids.dtype() == DType::I64 {
        ids.clone()
    } else {
        ids.to_dtype(DType::I64)?
    };
    let weight = weight.force_contiguous()?;
    let flat = ids.reshape((batch * seq_len,))?.force_contiguous()?;
    let gathered = weight.index_select(&flat, 0)?;
    Ok(gathered.reshape((batch, seq_len, hidden))?)
}
