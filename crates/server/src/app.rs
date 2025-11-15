use std::sync::Arc;

use anyhow::Result;
use deepseek_ocr_config::{AppConfig, LocalFileSystem};
use deepseek_ocr_core::{
    DecodeParameters, VisionSettings,
    runtime::{default_dtype_for_device, prepare_device_and_dtype},
};
use rocket::{Config, data::ToByteUnit};
use tracing::info;

use crate::{args::Args, routes, state::AppState};

pub async fn run(args: Args) -> Result<()> {
    let fs = LocalFileSystem::new("deepseek-ocr");
    let (mut app_config, descriptor) = AppConfig::load_or_init(&fs, args.config.as_deref())?;
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

    let vision_settings = VisionSettings {
        base_size: app_config.inference.base_size,
        image_size: app_config.inference.image_size,
        crop_mode: app_config.inference.crop_mode,
    };
    let decode_defaults = DecodeParameters {
        max_new_tokens: app_config.inference.max_new_tokens,
        do_sample: app_config.inference.do_sample,
        temperature: app_config.inference.temperature,
        top_p: if app_config.inference.top_p < 1.0 {
            Some(app_config.inference.top_p)
        } else {
            None
        },
        top_k: app_config.inference.top_k,
        repetition_penalty: app_config.inference.repetition_penalty,
        no_repeat_ngram_size: app_config.inference.no_repeat_ngram_size,
        seed: app_config.inference.seed,
        use_cache: app_config.inference.use_cache,
    };

    let state = AppState::bootstrap(
        fs.clone(),
        Arc::new(app_config.clone()),
        device.clone(),
        dtype,
        vision_settings,
        decode_defaults,
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
