use std::fs;

use anyhow::{Context, Result, anyhow};

use crate::args::InferArgs;

pub fn load_prompt(args: &InferArgs) -> Result<String> {
    if let Some(path) = &args.prompt_file {
        return fs::read_to_string(path)
            .with_context(|| format!("failed to read prompt file {}", path.display()))
            .map(|s| s.trim_end().to_owned());
    }
    if let Some(prompt) = &args.prompt {
        return Ok(prompt.clone());
    }
    Err(anyhow!(
        "prompt is required (use --prompt or --prompt-file)"
    ))
}
