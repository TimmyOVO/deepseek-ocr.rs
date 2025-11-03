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
    let args = Args::parse();
    logging::init(args.quiet);
    if let Err(err) = try_run(args) {
        error!(error = %err, "CLI failed");
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn try_run(args: Args) -> Result<()> {
    app::run(args)
}
