//! Utilities for streaming token-by-token outputs while preserving UTF-8 integrity.

/// Computes the suffix of `current` that differs from `previous`.
pub fn extract_delta(previous: &str, current: &str) -> String {
    if current.starts_with(previous) {
        return current[previous.len()..].to_owned();
    }

    let mut prefix_bytes = 0;
    for (a, b) in previous.chars().zip(current.chars()) {
        if a != b {
            break;
        }
        prefix_bytes += a.len_utf8();
    }

    current[prefix_bytes..].to_owned()
}

/// Tracks previously emitted text to compute new streaming deltas.
#[derive(Debug, Default, Clone)]
pub struct DeltaTracker {
    previous: String,
}

impl DeltaTracker {
    /// Creates a new tracker with an empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resets the tracker, clearing any remembered text.
    pub fn reset(&mut self) {
        self.previous.clear();
    }

    /// Returns the text that should be emitted for the current decode.
    ///
    /// When `is_final` is false, the tracker suppresses trailing `�` (replacement
    /// character) fragments so that callers stream only complete UTF-8 content.
    /// On the final call, the full decoded text is allowed through so the final
    /// output matches the model's prediction.
    pub fn advance(&mut self, current: &str, is_final: bool) -> String {
        let mut raw_delta = extract_delta(&self.previous, current);

        if raw_delta.is_empty() {
            self.previous = current.to_owned();
            return raw_delta;
        }

        if !is_final {
            if let Some(idx) = raw_delta.find(char::REPLACEMENT_CHARACTER) {
                if idx == 0 {
                    return String::new();
                }
                raw_delta.truncate(idx);
                self.previous.push_str(&raw_delta);
                return raw_delta;
            }
        }

        self.previous = current.to_owned();
        raw_delta
    }

    /// Returns the full text seen so far.
    pub fn snapshot(&self) -> &str {
        &self.previous
    }
}
