use std::time::Duration;

/// Value recorded for a benchmark field.
#[derive(Debug, Clone)]
pub enum BenchValue {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Text(String),
}

impl From<usize> for BenchValue {
    fn from(value: usize) -> Self {
        BenchValue::U64(value as u64)
    }
}

impl From<u64> for BenchValue {
    fn from(value: u64) -> Self {
        BenchValue::U64(value)
    }
}

impl From<i64> for BenchValue {
    fn from(value: i64) -> Self {
        BenchValue::I64(value)
    }
}

impl From<f64> for BenchValue {
    fn from(value: f64) -> Self {
        BenchValue::F64(value)
    }
}

impl From<bool> for BenchValue {
    fn from(value: bool) -> Self {
        BenchValue::Bool(value)
    }
}

impl From<&str> for BenchValue {
    fn from(value: &str) -> Self {
        BenchValue::Text(value.to_owned())
    }
}

impl From<String> for BenchValue {
    fn from(value: String) -> Self {
        BenchValue::Text(value)
    }
}

/// Key/value pair describing additional context for an event.
#[derive(Debug, Clone)]
pub struct BenchField {
    pub key: &'static str,
    pub value: BenchValue,
}

/// Single measurement emitted while benchmarks are enabled.
#[derive(Debug, Clone)]
pub struct BenchEvent {
    pub stage: &'static str,
    pub duration: Duration,
    pub fields: Vec<BenchField>,
}

impl BenchEvent {
    pub fn add_field<K, V>(&mut self, key: K, value: V)
    where
        K: Into<&'static str>,
        V: Into<BenchValue>,
    {
        self.fields.push(BenchField {
            key: key.into(),
            value: value.into(),
        });
    }
}

#[cfg(feature = "bench-metrics")]
mod imp {
    use super::{BenchEvent, BenchField};
    use once_cell::sync::Lazy;
    use std::sync::{Arc, RwLock};
    use std::time::{Duration, Instant};

    static RECORDER: Lazy<RwLock<Option<Arc<dyn BenchRecorder + Send + Sync>>>> =
        Lazy::new(|| RwLock::new(None));

    pub fn set_recorder(recorder: Option<Arc<dyn BenchRecorder + Send + Sync>>) {
        let mut slot = RECORDER
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        *slot = recorder;
    }

    pub struct Timer {
        stage: &'static str,
        start: Instant,
        emitted: bool,
    }

    impl Timer {
        pub fn new(stage: &'static str) -> Self {
            Self {
                stage,
                start: Instant::now(),
                emitted: false,
            }
        }

        pub fn finish<F>(mut self, update: F)
        where
            F: FnOnce(&mut BenchEvent),
        {
            if self.emitted {
                return;
            }
            self.emitted = true;
            if let Some(recorder) = current_recorder() {
                let mut event = BenchEvent {
                    stage: self.stage,
                    duration: self.start.elapsed(),
                    fields: Vec::new(),
                };
                update(&mut event);
                recorder.record(event);
            }
        }

        pub fn cancel(mut self) {
            self.emitted = true;
        }
    }

    impl Drop for Timer {
        fn drop(&mut self) {
            if self.emitted {
                return;
            }
            if let Some(recorder) = current_recorder() {
                let event = BenchEvent {
                    stage: self.stage,
                    duration: self.start.elapsed(),
                    fields: Vec::new(),
                };
                recorder.record(event);
            }
        }
    }

    fn current_recorder() -> Option<Arc<dyn BenchRecorder + Send + Sync>> {
        RECORDER
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().map(Arc::clone))
    }

    pub trait BenchRecorder {
        fn record(&self, event: BenchEvent);
    }

    pub fn record_instant(stage: &'static str, fields: impl IntoIterator<Item = BenchField>) {
        if let Some(recorder) = current_recorder() {
            let event = BenchEvent {
                stage,
                duration: Duration::default(),
                fields: fields.into_iter().collect(),
            };
            recorder.record(event);
        }
    }
}

#[cfg(not(feature = "bench-metrics"))]
mod imp {
    use super::{BenchEvent, BenchField};
    use std::sync::Arc;

    pub fn set_recorder(_recorder: Option<Arc<dyn BenchRecorder + Send + Sync>>) {}

    pub struct Timer;

    impl Timer {
        pub fn new(_stage: &'static str) -> Self {
            Self
        }

        pub fn finish<F>(self, _update: F)
        where
            F: FnOnce(&mut BenchEvent),
        {
        }

        pub fn cancel(self) {}
    }

    pub trait BenchRecorder {
        fn record(&self, _event: BenchEvent);
    }

    pub fn record_instant(_stage: &'static str, _fields: impl IntoIterator<Item = BenchField>) {}
}

pub use imp::{BenchRecorder, Timer, record_instant, set_recorder};
