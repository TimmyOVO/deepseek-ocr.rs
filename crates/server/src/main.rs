#![allow(clippy::too_many_arguments)]

#[macro_use]
extern crate rocket;

mod app;
mod args;
mod cors;
mod error;
mod generation;
mod logging;
mod models;
mod resources;
mod routes;
mod state;
mod stream;

use anyhow::Result;
use clap::Parser;
use tracing::error;

use crate::args::Args;

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
