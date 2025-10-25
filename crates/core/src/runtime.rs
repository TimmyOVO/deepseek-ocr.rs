use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::ValueEnum;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum DeviceKind {
    Cpu,
    Metal,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Precision {
    F32,
    F16,
    Bf16,
}

pub fn prepare_device_and_dtype(
    device: DeviceKind,
    precision: Option<Precision>,
) -> Result<(Device, Option<DType>)> {
    let (device, default_precision) = match device {
        DeviceKind::Cpu => (Device::Cpu, None),
        DeviceKind::Metal => (
            Device::new_metal(0).context("failed to initialise Metal device")?,
            Some(Precision::F16),
        ),
    };
    let dtype = precision.or(default_precision).map(dtype_from_precision);
    Ok((device, dtype))
}

pub fn default_dtype_for_device(device: &Device) -> DType {
    if device.is_metal() {
        DType::F16
    } else {
        DType::F32
    }
}

pub fn dtype_from_precision(p: Precision) -> DType {
    match p {
        Precision::F32 => DType::F32,
        Precision::F16 => DType::F16,
        Precision::Bf16 => DType::BF16,
    }
}
