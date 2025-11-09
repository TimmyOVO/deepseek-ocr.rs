use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use hf_hub::api::sync::Api;

use super::AssetProvider;
use crate::{DEFAULT_REPO_ID, copy_to_target};

pub(crate) struct HuggingFaceProvider;

impl AssetProvider for HuggingFaceProvider {
    fn display_name(&self) -> &'static str {
        "Hugging Face Hub"
    }

    fn download(&self, repo_id: &str, remote_name: &str, target: &Path) -> Result<PathBuf> {
        let api = Api::new().context("failed to initialise Hugging Face API client")?;
        let repo = api.model(repo_id.to_string());
        let cached = repo
            .get(remote_name)
            .with_context(|| format!("failed to download {remote_name} from Hugging Face"))?;

        copy_to_target(&cached, target)?;
        Ok(target.to_path_buf())
    }

    fn benchmark(&self) -> Option<Duration> {
        let start = Instant::now();
        let api = Api::new().ok()?;
        let repo = api.model(DEFAULT_REPO_ID.to_string());
        repo.info().ok()?;
        Some(start.elapsed())
    }
}
