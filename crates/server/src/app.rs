use std::sync::Arc;

use anyhow::Result;
use deepseek_ocr_config::{AppConfig, ConfigOverrides, LocalFileSystem};
use deepseek_ocr_core::runtime::{default_dtype_for_device, prepare_device_and_dtype};
use rocket::{Config, data::ToByteUnit};
use tracing::info;

use crate::{args::Args, routes, state::AppState};

pub async fn run(args: Args) -> Result<()> {
    let fs = LocalFileSystem::new("deepseek-ocr");
    let (mut app_config, descriptor) = AppConfig::load_or_init(&fs, args.config.as_deref())?;
    let base_inference = app_config.inference.clone();
    app_config += &args;
    app_config.normalise(&fs)?;
    info!(
        "Using configuration {} (active model `{}`)",
        descriptor.location.display_with(&fs)?,
        app_config.models.active
    );

    let (device, maybe_dtype) =
        prepare_device_and_dtype(app_config.inference.device, app_config.inference.precision)?;
    let dtype = maybe_dtype.unwrap_or_else(|| default_dtype_for_device(&device));

    let inference_overrides = ConfigOverrides::from(&args).inference;

    let state = AppState::bootstrap(
        fs.clone(),
        Arc::new(app_config.clone()),
        device.clone(),
        dtype,
        base_inference,
        inference_overrides,
    )?;

    let figment = Config::figment()
        .merge(("port", app_config.server.port))
        .merge(("address", app_config.server.host.clone()))
        .merge((
            "limits",
            rocket::data::Limits::default()
                .limit("json", 50.megabytes())
                .limit("bytes", 50.megabytes()),
        ));

    info!(
        "Server ready on {}:{}",
        app_config.server.host, app_config.server.port,
    );

    rocket::custom(figment)
        .attach(crate::cors::Cors)
        .manage(state)
        .mount("/v1", routes::v1_routes())
        .launch()
        .await
        .map_err(|err| anyhow::anyhow!("rocket failed: {err}"))?;

    Ok(())
}
