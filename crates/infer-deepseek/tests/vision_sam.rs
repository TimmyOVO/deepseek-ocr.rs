mod common;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use common::test_utils::{with_shared_ocr_model, workspace_path};
use deepseek_ocr_infer_deepseek::{
    config::load_ocr_config,
    model::DeepseekOcrModel,
    vision::sam::{SamBackbone, window_partition_shape, window_unpartition_shape},
};

fn with_model<F>(label: &str, f: F) -> Result<()>
where
    F: FnOnce(&DeepseekOcrModel) -> Result<()>,
{
    match with_shared_ocr_model(f) {
        Ok(result) => Ok(result),
        Err(err) => {
            eprintln!("skipping {label}: {err}");
            Ok(())
        }
    }
}

#[test]
fn sam_params_match_hf_config() {
    let cfg_path = workspace_path("DeepSeek-OCR/config.json");
    let cfg = load_ocr_config(Some(&cfg_path)).expect("hf config must load");
    let sam = SamBackbone::with_dummy_weights(&cfg).expect("sam params");
    assert_eq!(sam.params.image_size, 1024);
    assert_eq!(sam.params.patch_size, 16);
    assert_eq!(sam.params.embed_dim, 768);
    assert_eq!(sam.params.depth, 12);
    assert_eq!(sam.params.num_heads, 12);
    assert_eq!(sam.params.global_attn_indexes, vec![2, 5, 8, 11]);
}

#[test]
fn sam_forward_shape_check() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    let cfg_path = workspace_path("DeepSeek-OCR/config.json");
    let cfg = load_ocr_config(Some(&cfg_path)).expect("hf config must load");
    let sam = SamBackbone::with_dummy_weights(&cfg).expect("sam params");

    let device = Device::Cpu;
    let input = Tensor::zeros(
        (1, 3, sam.params.image_size, sam.params.image_size),
        DType::F32,
        &device,
    )
    .expect("allocate input tensor");
    let output = sam.forward(&input).expect("sam forward shape");

    let (batch, channels, h, w) = output.shape().dims4().expect("output should be 4D (NCHW)");

    assert_eq!(batch, 1);
    assert_eq!(
        channels,
        *sam.params
            .out_channels
            .last()
            .expect("sam out_channels not empty")
    );
    let tokens_per_side = sam.params.image_size / sam.params.patch_size;
    assert_eq!(h, tokens_per_side / 4);
    assert_eq!(w, tokens_per_side / 4);
}

#[test]
fn window_partition_math() {
    let shape = window_partition_shape(64, 48, 14);
    assert_eq!(shape.padded_height, 70);
    assert_eq!(shape.padded_width, 56);
    assert_eq!(shape.tiles_h, 5);
    assert_eq!(shape.tiles_w, 4);

    let (h, w) = window_unpartition_shape(shape, 14);
    assert_eq!(h, 70);
    assert_eq!(w, 56);
}

#[test]
fn sam_real_weights_forward_shapes() -> Result<()> {
    with_model("SAM safetensor load test", |model| {
        let sam = model.sam_backbone();
        let device = Device::Cpu;
        let input = Tensor::zeros(
            (1, 3, sam.params.image_size, sam.params.image_size),
            DType::F32,
            &device,
        )
        .expect("input tensor");
        let output = sam.forward(&input)?;
        assert_eq!(output.dtype(), DType::F32);
        Ok(())
    })
}
