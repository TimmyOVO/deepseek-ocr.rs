use deepseek_ocr_dsq::{DsqBiasDType, DsqRecord, DsqTensorDType};

fn format_bias(record: &DsqRecord) -> String {
    match (record.bias_dtype, record.bias_len) {
        (Some(dtype), Some(len)) => format!("{dtype} ({} bytes)", len),
        (None, None) => "none".to_string(),
        _ => "invalid metadata".to_string(),
    }
}

#[derive(Clone, Copy, Debug)]
enum QuantDTypeArg {
    Q8_0,
    Q4K,
    Q6K,
}

impl QuantDTypeArg {
    fn to_dtype(self) -> DsqTensorDType {
        match self {
            Self::Q8_0 => DsqTensorDType::Q8_0,
            Self::Q4K => DsqTensorDType::Q4K,
            Self::Q6K => DsqTensorDType::Q6K,
        }
    }
}

#[test]
fn bias_formatter_handles_all_cases() {
    let base = DsqRecord {
        name: "layer".to_string(),
        out_dim: 1,
        in_dim: 1,
        q_dtype: DsqTensorDType::Q8_0,
        q_offset: 0,
        q_len: 10,
        bias_offset: None,
        bias_len: None,
        bias_dtype: None,
    };
    assert_eq!(format_bias(&base), "none");

    let mut with_bias = base.clone();
    with_bias.bias_dtype = Some(DsqBiasDType::F32);
    with_bias.bias_len = Some(128);
    assert_eq!(format_bias(&with_bias), "F32 (128 bytes)");

    let mut invalid = base;
    invalid.bias_dtype = Some(DsqBiasDType::F16);
    assert_eq!(format_bias(&invalid), "invalid metadata");
}

#[test]
fn quant_dtype_arg_maps_variants() {
    assert_eq!(QuantDTypeArg::Q8_0.to_dtype(), DsqTensorDType::Q8_0);
    assert_eq!(QuantDTypeArg::Q4K.to_dtype(), DsqTensorDType::Q4K);
    assert_eq!(QuantDTypeArg::Q6K.to_dtype(), DsqTensorDType::Q6K);
}
