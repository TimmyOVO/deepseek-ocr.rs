use anyhow::Result;
use deepseek_ocr_pipeline::deepseek_ocr_config::{AppConfig, ConfigOverrides, LocalFileSystem};
use deepseek_ocr_pipeline::OcrConfigPatch;
use rocket::{Config, data::ToByteUnit};
use tracing::info;

use crate::{args::Args, routes, state::AppState};

pub async fn run(args: Args) -> Result<()> {
    let fs = LocalFileSystem::new("deepseek-ocr");
    let (mut app_config, descriptor) = AppConfig::load_or_init(&fs, args.model.config.as_deref())?;
    let overrides = ConfigOverrides::from(&args);
    app_config.apply_overrides(&overrides);
    app_config.normalise(&fs)?;
    info!(
        "Using configuration {} (active model `{}`)",
        descriptor.location.display_with(&fs)?,
        app_config.models.active
    );

    let defaults_layer: Option<OcrConfigPatch> = None;
    let state = AppState::bootstrap(&args, defaults_layer)?;

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
