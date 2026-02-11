use candle_core::{DType, Device, Tensor};
use deepseek_ocr_infer_dots::quant::QuantLinear;

#[test]
fn quant_linear_forward_preserves_last_dim() -> anyhow::Result<()> {
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
