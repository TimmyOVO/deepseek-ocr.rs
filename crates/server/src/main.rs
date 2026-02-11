use anyhow::Result;
use clap::Parser;
use tracing::error;

use deepseek_ocr_server::{app, args::Args, logging};

#[rocket::main]
async fn main() -> Result<()> {
    logging::init();
    let args = Args::parse();
    match app::run(args).await {
        Ok(()) => Ok(()),
        Err(err) => {
            error!(error = %err, "Server failed");
            Err(err)
        }
    }
}
