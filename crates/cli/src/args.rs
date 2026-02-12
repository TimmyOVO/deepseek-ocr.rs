use std::path::PathBuf;

use clap::{Args as ClapArgs, Parser, Subcommand};
use deepseek_ocr_config::{
    CommonInferenceArgs, CommonModelArgs, ConfigOverrides, build_config_overrides,
};

use crate::debug::DebugArgs;

#[derive(Parser, Debug)]
#[command(author, version, about = "DeepSeek-OCR CLI", long_about = None)]
pub struct Cli {
    #[command(flatten)]
    pub infer: InferArgs,

    #[command(subcommand)]
    pub command: Option<CliCommand>,
}

#[derive(Subcommand, Debug)]
pub enum CliCommand {
    Weights(WeightsArgs),
}

#[derive(ClapArgs, Debug)]
pub struct WeightsArgs {
    #[command(subcommand)]
    pub command: WeightsCommand,
}

#[derive(Subcommand, Debug)]
pub enum WeightsCommand {
    #[command(name = "snapshot")]
    Snapshot(SnapshotArgs),
}

#[derive(ClapArgs, Debug)]
pub struct SnapshotArgs {
    #[arg(long, value_name = "PATH", help_heading = "Application")]
    pub config: Option<PathBuf>,

    #[arg(long = "in", value_name = "PATH", help_heading = "Snapshot")]
    pub input: PathBuf,

    #[arg(long = "out", value_name = "PATH", help_heading = "Snapshot")]
    pub output: PathBuf,

    #[arg(
        long,
        value_name = "DTYPE",
        default_value = "Q8_0",
        value_parser = ["Q8_0", "Q4_K", "Q6_K"],
        help_heading = "Snapshot"
    )]
    pub dtype: String,

    #[arg(long, value_name = "TARGETS", default_value = "text", value_parser = ["text", "text+projector"], help_heading = "Snapshot")]
    pub targets: String,
}

#[derive(ClapArgs, Debug)]
pub struct InferArgs {
    #[command(flatten)]
    pub model: CommonModelArgs,

    #[command(flatten)]
    pub inference: CommonInferenceArgs,

    /// Prompt text. Use `<image>` tokens to denote image slots.
    #[arg(long, conflicts_with = "prompt_file")]
    pub prompt: Option<String>,

    /// Prompt file path (UTF-8). Overrides `--prompt` when provided.
    #[arg(long, value_name = "PATH", conflicts_with = "prompt")]
    pub prompt_file: Option<PathBuf>,

    #[command(flatten)]
    pub debug: DebugArgs,

    /// Image files corresponding to `<image>` placeholders, in order.
    #[arg(long = "image", value_name = "PATH")]
    pub images: Vec<PathBuf>,

    /// Enable benchmark instrumentation (requires `bench-metrics` feature).
    #[arg(long, help_heading = "Benchmark")]
    pub bench: bool,

    /// Write benchmark events to a JSON file.
    #[arg(long, value_name = "PATH", help_heading = "Benchmark")]
    pub bench_output: Option<PathBuf>,

    /// Quiet mode - output only the final result without logs or progress.
    #[arg(short, long, help_heading = "Application")]
    pub quiet: bool,
}

impl From<&InferArgs> for ConfigOverrides {
    fn from(args: &InferArgs) -> Self {
        build_config_overrides(&args.model, &args.inference, None)
    }
}
