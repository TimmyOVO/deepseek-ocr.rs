use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use indicatif::{HumanBytes, ProgressBar, ProgressStyle, style::ProgressTracker};

const PROGRESS_TEMPLATE: &str =
    "{msg} [{elapsed_precise}] [{wide_bar}] {bytes}/{total_bytes} {bytes_per_sec_smoothed} ({eta})";
const PROGRESS_LABEL_MAX: usize = 30;

#[derive(Clone)]
pub(crate) struct SmoothedRate {
    window: Duration,
    samples: VecDeque<(Instant, u64)>,
}

impl Default for SmoothedRate {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(1),
            samples: VecDeque::new(),
        }
    }
}

impl ProgressTracker for SmoothedRate {
    fn clone_box(&self) -> Box<dyn ProgressTracker> {
        Box::new(self.clone())
    }

    fn tick(&mut self, state: &indicatif::ProgressState, now: Instant) {
        if let Some((last, _)) = self.samples.back() {
            if now.duration_since(*last) < Duration::from_millis(20) {
                return;
            }
        }

        self.samples.push_back((now, state.pos()));
        while let Some((time, _)) = self.samples.front() {
            if now.duration_since(*time) > self.window {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    fn reset(&mut self, _state: &indicatif::ProgressState, _now: Instant) {
        self.samples.clear();
    }

    fn write(&self, _state: &indicatif::ProgressState, w: &mut dyn std::fmt::Write) {
        if let (Some((t0, p0)), Some((t1, p1))) = (self.samples.front(), self.samples.back()) {
            if self.samples.len() > 1 && t1 > t0 {
                let elapsed = t1.duration_since(*t0).as_millis() as f64 / 1000.0;
                let bytes = (p1 - p0) as f64;
                let rate = if elapsed > 0.0 { bytes / elapsed } else { 0.0 };
                let _ = write!(w, "{}/s", HumanBytes(rate as u64));
                return;
            }
        }

        let _ = write!(w, "-");
    }
}

pub(crate) fn create_progress_bar(total: u64, label: &str) -> ProgressBar {
    let bar = ProgressBar::new(total);
    let style = ProgressStyle::with_template(PROGRESS_TEMPLATE)
        .expect("progress template should be valid")
        .with_key("bytes_per_sec_smoothed", SmoothedRate::default());
    bar.set_style(style);
    let message = if label.len() > PROGRESS_LABEL_MAX {
        format!("..{}", &label[label.len() - PROGRESS_LABEL_MAX..])
    } else {
        label.to_string()
    };
    bar.set_message(message);
    bar
}
