mod app;
mod args;
mod bench;
mod logging;
mod prompt;
mod resources;

use crate::args::Args;
use anyhow::Result;
use clap::Parser;
use tracing::error;

fn main() {
    logging::init();
    if let Err(err) = try_run() {
        error!(error = %err, "CLI failed");
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn try_run() -> Result<()> {
    let args = Args::parse();
    app::run(args)
}
