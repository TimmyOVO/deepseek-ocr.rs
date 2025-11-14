mod app;
mod args;
mod bench;
mod logging;
mod prompt;
mod resources;

use crate::args::{Cli, CliCommand};
use anyhow::Result;
use clap::Parser;
use tracing::error;

fn main() {
    let cli = Cli::parse();
    logging::init(cli.infer.quiet);
    if let Err(err) = try_run(cli) {
        error!(error = %err, "CLI failed");
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn try_run(cli: Cli) -> Result<()> {
    let Cli { infer, command } = cli;
    match command {
        Some(CliCommand::Weights(weights)) => app::run_weights(weights),
        None => app::run_inference(infer),
    }
}
