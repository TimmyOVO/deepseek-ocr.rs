use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use deepseek_ocr_config::{AppConfig, LocalFileSystem};
use deepseek_ocr_core::{
    model::DeepseekOcrModel,
    runtime::{default_dtype_for_device, prepare_device_and_dtype},
};
use rocket::{Config, data::ToByteUnit};
use tokenizers::Tokenizer;
use tracing::info;

use crate::{
    args::Args,
    resources::{ensure_config_file, ensure_tokenizer_file, prepare_weights_path},
    routes,
    state::AppState,
};

pub async fn run(args: Args) -> Result<()> {
    let fs = LocalFileSystem::new("deepseek-ocr");
    let (mut app_config, descriptor) = AppConfig::load_or_init(&fs, args.config.as_deref())?;
    app_config += &args;
    app_config.normalise(&fs)?;
    let resources = app_config.active_model_resources(&fs)?;

    info!(
        "Using configuration {} (active model `{}`)",
        descriptor.location.display_with(&fs)?,
        app_config.models.active
    );

    let config_path = ensure_config_file(&fs, &resources.config)?;
    let tokenizer_path = ensure_tokenizer_file(&fs, &resources.tokenizer)?;
    let weights_path = prepare_weights_path(&fs, &resources.weights)?;

    let (device, maybe_dtype) =
        prepare_device_and_dtype(app_config.inference.device, app_config.inference.precision)?;
    let dtype = maybe_dtype.unwrap_or_else(|| default_dtype_for_device(&device));

    let model = DeepseekOcrModel::load(Some(&config_path), Some(&weights_path), device, dtype)
        .context("failed to load DeepSeek-OCR model")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
        anyhow::anyhow!(
            "failed to load tokenizer from {}: {err}",
            tokenizer_path.display()
        )
    })?;

    let state = AppState::new(
        Arc::new(Mutex::new(model)),
        Arc::new(tokenizer),
        app_config.inference.base_size,
        app_config.inference.image_size,
        app_config.inference.crop_mode,
        app_config.inference.max_new_tokens,
        app_config.inference.use_cache,
        app_config.inference.do_sample,
        app_config.inference.temperature,
        app_config.inference.top_p,
        app_config.inference.top_k,
        app_config.inference.repetition_penalty,
        app_config.inference.no_repeat_ngram_size,
        app_config.inference.seed,
        app_config.server.model_id.clone(),
    );

    let model_id = state.model_id.clone();

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
        "Server ready on {}:{} ({model_id})",
        app_config.server.host, app_config.server.port
    );

    rocket::custom(figment)
        .manage(state)
        .mount("/v1", routes::v1_routes())
        .launch()
        .await
        .map_err(|err| anyhow::anyhow!("rocket failed: {err}"))?;

    Ok(())
}
