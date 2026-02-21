use std::path::Path;

#[cfg(feature = "cli-debug")]
use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

/// Additional debug/test-only CLI flags.
///
/// When `cli-debug` is disabled, this struct is empty so the flags disappear
/// from `--help` and the rest of the CLI code stays unchanged.
#[cfg(feature = "cli-debug")]
#[derive(Args, Debug, Default)]
pub struct DebugArgs {
    /// Baseline prompt.json path; load the `rendered_prompt` field.
    #[arg(
        long,
        value_name = "PATH",
        help_heading = "Debug",
        conflicts_with_all = ["prompt", "prompt_file"]
    )]
    prompt_json: Option<PathBuf>,

    /// Write inference artifacts to a JSON file.
    #[arg(long, value_name = "PATH", help_heading = "Debug")]
    output_json: Option<PathBuf>,
}

#[cfg(not(feature = "cli-debug"))]
#[derive(Args, Debug, Default)]
pub struct DebugArgs {}

pub struct DebugOutput<'a> {
    pub model_id: &'a str,
    pub weights_path: &'a Path,
    pub tokenizer_path: &'a Path,
    pub device: &'a str,
    pub dtype: &'a str,
    pub template: &'a str,
    pub base_size: u32,
    pub image_size: u32,
    pub crop_mode: bool,
    pub max_new_tokens: usize,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: Option<usize>,
    pub use_cache: bool,
    pub prompt_user: &'a str,
    pub rendered_prompt: &'a str,
    pub image_paths: &'a [String],
    pub prompt_tokens: usize,
    pub generated_len: usize,
    pub tokens: &'a [i64],
    pub decoded: &'a str,
    pub normalized: &'a str,
}

pub fn load_prompt_override(args: &DebugArgs) -> Result<Option<String>> {
    #[cfg(feature = "cli-debug")]
    {
        use anyhow::Context;

        let Some(path) = args.prompt_json.as_deref() else {
            return Ok(None);
        };

        #[derive(serde::Deserialize)]
        struct PromptJson {
            rendered_prompt: String,
        }

        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read prompt json {}", path.display()))?;
        let parsed: PromptJson = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse prompt json {}", path.display()))?;
        Ok(Some(parsed.rendered_prompt))
    }

    #[cfg(not(feature = "cli-debug"))]
    {
        let _ = args;
        Ok(None)
    }
}

pub fn wants_output_json(args: &DebugArgs) -> bool {
    #[cfg(feature = "cli-debug")]
    {
        args.output_json.is_some()
    }

    #[cfg(not(feature = "cli-debug"))]
    {
        let _ = args;
        false
    }
}

pub fn write_output_json(args: &DebugArgs, out: DebugOutput<'_>) -> Result<()> {
    #[cfg(feature = "cli-debug")]
    {
        use anyhow::Context;

        let Some(path) = args.output_json.as_deref() else {
            return Ok(());
        };

        #[derive(serde::Serialize)]
        struct CliOutputJson<'a> {
            schema_version: u32,
            model_id: &'a str,
            weights: String,
            tokenizer: String,
            device: &'a str,
            dtype: &'a str,
            template: &'a str,
            base_size: u32,
            image_size: u32,
            crop_mode: bool,
            max_new_tokens: usize,
            repetition_penalty: f32,
            no_repeat_ngram_size: Option<usize>,
            use_cache: bool,
            prompt: &'a str,
            rendered_prompt: &'a str,
            image_paths: &'a [String],
            prompt_tokens: usize,
            generated_len: usize,
            tokens: &'a [i64],
            decoded: &'a str,
            normalized: &'a str,
        }

        let json = CliOutputJson {
            schema_version: 1,
            model_id: out.model_id,
            weights: out.weights_path.display().to_string(),
            tokenizer: out.tokenizer_path.display().to_string(),
            device: out.device,
            dtype: out.dtype,
            template: out.template,
            base_size: out.base_size,
            image_size: out.image_size,
            crop_mode: out.crop_mode,
            max_new_tokens: out.max_new_tokens,
            repetition_penalty: out.repetition_penalty,
            no_repeat_ngram_size: out.no_repeat_ngram_size,
            use_cache: out.use_cache,
            prompt: out.prompt_user,
            rendered_prompt: out.rendered_prompt,
            image_paths: out.image_paths,
            prompt_tokens: out.prompt_tokens,
            generated_len: out.generated_len,
            tokens: out.tokens,
            decoded: out.decoded,
            normalized: out.normalized,
        };

        let bytes = serde_json::to_vec_pretty(&json).context("serialize output json")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
        std::fs::write(path, bytes)
            .with_context(|| format!("failed to write output json {}", path.display()))?;
        Ok(())
    }

    #[cfg(not(feature = "cli-debug"))]
    {
        let _ = args;
        let DebugOutput {
            model_id,
            weights_path,
            tokenizer_path,
            device,
            dtype,
            template,
            base_size,
            image_size,
            crop_mode,
            max_new_tokens,
            repetition_penalty,
            no_repeat_ngram_size,
            use_cache,
            prompt_user,
            rendered_prompt,
            image_paths,
            prompt_tokens,
            generated_len,
            tokens,
            decoded,
            normalized,
        } = out;
        let _ = (
            model_id,
            weights_path,
            tokenizer_path,
            device,
            dtype,
            template,
            base_size,
            image_size,
            crop_mode,
            max_new_tokens,
            repetition_penalty,
            no_repeat_ngram_size,
            use_cache,
            prompt_user,
            rendered_prompt,
            image_paths,
            prompt_tokens,
            generated_len,
            tokens,
            decoded,
            normalized,
        );
        Ok(())
    }
}
