use anyhow::{Result, anyhow, ensure};
use tokenizers::Tokenizer;

pub const IMAGE_PLACEHOLDER: &str = "<image>";

#[derive(Debug, Clone, Copy)]
pub struct PromptBuildOptions<'a> {
    pub placeholder: &'a str,
    pub image_label: &'a str,
    pub prefix_tokens: &'a [i64],
    pub suffix_tokens: &'a [i64],
}

impl<'a> PromptBuildOptions<'a> {
    pub fn image_slots(image_label: &'a str) -> Self {
        Self {
            placeholder: IMAGE_PLACEHOLDER,
            image_label,
            prefix_tokens: &[],
            suffix_tokens: &[],
        }
    }

    pub fn with_prefix(mut self, prefix_tokens: &'a [i64]) -> Self {
        self.prefix_tokens = prefix_tokens;
        self
    }

    pub fn with_suffix(mut self, suffix_tokens: &'a [i64]) -> Self {
        self.suffix_tokens = suffix_tokens;
        self
    }
}

#[derive(Debug, Clone)]
pub struct PromptTokenSequence {
    pub tokens: Vec<i64>,
    pub image_mask: Vec<u8>,
}

impl PromptTokenSequence {
    pub fn image_token_count(&self) -> usize {
        self.image_mask.iter().filter(|&&flag| flag != 0).count()
    }
}

pub fn grid_token_count(grid: (usize, usize, usize), merge_size: usize) -> Result<usize> {
    ensure!(merge_size > 0, "merge size must be positive");
    let (t, h, w) = grid;
    ensure!(
        h % merge_size == 0 && w % merge_size == 0,
        "grid {:?} not divisible by merge size {}",
        grid,
        merge_size
    );
    Ok(t * (h / merge_size) * (w / merge_size))
}

pub fn build_prompt_tokens_with<F>(
    tokenizer: &Tokenizer,
    prompt: &str,
    image_count: usize,
    options: PromptBuildOptions<'_>,
    mut append_image_tokens: F,
) -> Result<PromptTokenSequence>
where
    F: FnMut(usize, &mut Vec<i64>, &mut Vec<u8>) -> Result<()>,
{
    let segments: Vec<&str> = prompt.split(options.placeholder).collect();
    let slots = segments.len().saturating_sub(1);
    ensure!(
        slots == image_count,
        "prompt/image mismatch: {slots} slots vs {image_count} {}",
        options.image_label
    );

    let mut tokens = Vec::new();
    let mut image_mask = Vec::new();
    tokens.extend_from_slice(options.prefix_tokens);
    image_mask.extend(std::iter::repeat_n(0u8, options.prefix_tokens.len()));

    for (idx, segment) in segments.iter().enumerate() {
        if !segment.is_empty() {
            let encoding = tokenizer
                .encode(*segment, false)
                .map_err(|err| anyhow!("tokenization failed at segment {idx}: {err}"))?;
            tokens.extend(encoding.get_ids().iter().map(|&id| i64::from(id)));
            image_mask.extend(std::iter::repeat_n(0u8, encoding.len()));
        }

        if idx < image_count {
            append_image_tokens(idx, &mut tokens, &mut image_mask)?;
        }
    }

    tokens.extend_from_slice(options.suffix_tokens);
    image_mask.extend(std::iter::repeat_n(0u8, options.suffix_tokens.len()));
    ensure!(
        tokens.len() == image_mask.len(),
        "token/mask length mismatch: {} vs {}",
        tokens.len(),
        image_mask.len()
    );

    Ok(PromptTokenSequence { tokens, image_mask })
}
