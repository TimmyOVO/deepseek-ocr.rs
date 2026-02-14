use anyhow::{Result, anyhow, ensure};
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

/// Concatenates per-image embeddings to a single `[tokens, hidden]` tensor.
pub fn concat_token_embeddings(mut per_image: Vec<Tensor>) -> Result<Option<Tensor>> {
    if per_image.is_empty() {
        return Ok(None);
    }
    if per_image.len() == 1 {
        return Ok(per_image.pop());
    }
    let refs: Vec<&Tensor> = per_image.iter().collect();
    Ok(Some(Tensor::cat(&refs, 0)?))
}

/// Replaces masked token rows in `embeddings` with `per_batch` image embeddings.
///
/// - `embeddings`: `[batch, seq, hidden]`
/// - `mask`: `[batch, seq]` where non-zero marks rows to replace
/// - `per_batch`: one tensor per batch row, each `[masked_tokens, hidden]`
pub fn inject_embeddings_by_mask(
    embeddings: &Tensor,
    mask: &Tensor,
    per_batch: &[Tensor],
) -> Result<Tensor> {
    let (batch, seq_len, hidden) = embeddings.shape().dims3()?;
    let mask = to_dtype_if_needed(mask, DType::U8)?;
    ensure!(
        mask.shape().dims() == [batch, seq_len],
        "image mask must have shape [batch, seq]"
    );

    let mut rows = Vec::with_capacity(batch);
    for b in 0..batch {
        let row = embeddings
            .get(b)?
            .reshape((seq_len, hidden))?
            .contiguous()?;
        let mask_row = mask.get(b)?.reshape((seq_len,))?;
        let mask_vec = mask_row.to_vec1::<u8>()?;
        let ones = mask_vec.iter().filter(|&&flag| flag != 0).count();
        if ones == 0 {
            rows.push(row);
            continue;
        }

        let replacements = per_batch
            .get(b)
            .ok_or_else(|| anyhow!("missing image embeddings for batch {b}"))?;
        let replacements = to_dtype_if_needed(replacements, row.dtype())?;
        let replacements = to_device_if_needed(&replacements, row.device())?;
        let replacements = replacements.contiguous()?;
        let (rep_tokens, rep_hidden) = replacements.shape().dims2()?;
        ensure!(
            rep_tokens == ones,
            "image embeddings provide {rep_tokens} tokens but mask requires {ones}"
        );
        ensure!(
            rep_hidden == hidden,
            "image embeddings hidden size {rep_hidden} mismatches embedding hidden size {hidden}"
        );

        let mut rep_offset = 0usize;
        let mut cursor = 0usize;
        let mut segments = Vec::new();
        while cursor < seq_len {
            let flag = mask_vec[cursor];
            let start = cursor;
            while cursor < seq_len && mask_vec[cursor] == flag {
                cursor += 1;
            }
            let length = cursor - start;
            let segment = if flag == 0 {
                row.narrow(0, start, length)?
            } else {
                let seg = replacements.narrow(0, rep_offset, length)?;
                rep_offset += length;
                seg
            };
            segments.push(segment);
        }
        ensure!(
            rep_offset == ones,
            "not all replacement tokens were consumed (used {rep_offset} of {ones})"
        );
        let refs: Vec<&Tensor> = segments.iter().collect();
        rows.push(Tensor::cat(&refs, 0)?);
    }

    let refs: Vec<&Tensor> = rows.iter().collect();
    Ok(Tensor::stack(&refs, 0)?)
}
