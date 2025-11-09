use std::{
    cmp::Ordering,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::Result;

mod hugging_face;
mod modelscope;

pub(crate) use hugging_face::HuggingFaceProvider;
pub(crate) use modelscope::ModelScopeProvider;

pub(crate) trait AssetProvider: Sync {
    fn display_name(&self) -> &'static str;
    fn download(&self, repo_id: &str, remote_name: &str, target: &Path) -> Result<PathBuf>;
    fn benchmark(&self) -> Option<Duration>;
}

static HUGGING_FACE_PROVIDER: HuggingFaceProvider = HuggingFaceProvider;
static MODELSCOPE_PROVIDER: ModelScopeProvider = ModelScopeProvider;
static PROVIDERS: [&'static dyn AssetProvider; 2] = [&HUGGING_FACE_PROVIDER, &MODELSCOPE_PROVIDER];

pub(crate) fn providers_in_download_order() -> Vec<&'static dyn AssetProvider> {
    let mut measured: Vec<(&'static dyn AssetProvider, Option<Duration>)> = PROVIDERS
        .iter()
        .copied()
        .map(|provider| (provider, provider.benchmark()))
        .collect();

    measured.sort_by(|a, b| match (&a.1, &b.1) {
        (Some(la), Some(lb)) => la.cmp(lb),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    });

    measured.into_iter().map(|(provider, _)| provider).collect()
}

pub(crate) fn announce_provider(
    provider: &dyn AssetProvider,
    repo_id: &str,
    remote: &str,
    target: &Path,
) {
    tracing::info!(
        "Downloading {remote} from {repo_id} via {} -> {}",
        provider.display_name(),
        target.display()
    );
}
