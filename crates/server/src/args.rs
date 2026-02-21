use clap::Parser;
use deepseek_ocr_pipeline::deepseek_ocr_config::{
    build_config_overrides, CommonInferenceArgs, CommonModelArgs, ConfigOverrides, ServerBindArgs,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "DeepSeek-OCR API Server", long_about = None)]
pub struct Args {
    #[command(flatten)]
    pub model: CommonModelArgs,

    #[command(flatten)]
    pub inference: CommonInferenceArgs,

    #[command(flatten)]
    pub bind: ServerBindArgs,
}

impl From<&Args> for ConfigOverrides {
    fn from(args: &Args) -> Self {
        build_config_overrides(&args.model, &args.inference, Some(&args.bind))
    }
}
