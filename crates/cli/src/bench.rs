use std::{path::PathBuf, time::Duration};

#[cfg(feature = "bench-metrics")]
use std::{collections::HashMap, fs};

#[cfg(feature = "bench-metrics")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "bench-metrics")]
use anyhow::Context;
use anyhow::Result;
#[cfg(not(feature = "bench-metrics"))]
use anyhow::bail;
use deepseek_ocr_core::benchmark::BenchEvent;
#[cfg(feature = "bench-metrics")]
use deepseek_ocr_core::benchmark::BenchRecorder;
#[cfg(feature = "bench-metrics")]
use deepseek_ocr_core::benchmark::BenchValue;
#[cfg(feature = "bench-metrics")]
use serde_json::json;
use tracing::info;

#[cfg(feature = "bench-metrics")]
use deepseek_ocr_core::benchmark::set_recorder;

#[derive(Debug, Clone)]
pub struct StageTotal {
    pub stage: String,
    pub count: usize,
    pub total: Duration,
    pub min: Duration,
    pub max: Duration,
}

#[derive(Debug, Clone)]
pub struct Report {
    pub events: Vec<BenchEvent>,
    pub stage_totals: Vec<StageTotal>,
    pub output_path: Option<PathBuf>,
}

#[cfg(feature = "bench-metrics")]
pub struct Session {
    collector: Arc<Collector>,
    output: Option<PathBuf>,
    finished: bool,
}

#[cfg(not(feature = "bench-metrics"))]
pub struct Session {
    _phantom: (),
}

#[cfg(feature = "bench-metrics")]
impl Session {
    fn new(output: Option<PathBuf>) -> Result<Self> {
        let collector = Arc::new(Collector::default());
        set_recorder(Some(collector.clone()));
        Ok(Self {
            collector,
            output,
            finished: false,
        })
    }

    pub fn finalize(mut self) -> Result<Report> {
        set_recorder(None);
        self.finished = true;
        let events = self.collector.snapshot();
        if let Some(path) = &self.output {
            write_report(path, &events)?;
        }
        Ok(Report {
            stage_totals: totals_for_events(&events),
            events,
            output_path: self.output.clone(),
        })
    }
}

#[cfg(feature = "bench-metrics")]
impl Drop for Session {
    fn drop(&mut self) {
        if !self.finished {
            set_recorder(None);
        }
    }
}

#[cfg(not(feature = "bench-metrics"))]
impl Session {
    fn new(_output: Option<PathBuf>) -> Result<Self> {
        bail!("Benchmarking requires compiling with --features bench-metrics on deepseek-ocr-core");
    }

    #[allow(unused)]
    pub fn finalize(self) -> Result<Report> {
        unreachable!("benchmark session cannot finalize without bench-metrics feature")
    }
}

pub fn maybe_start(enabled: bool, output: Option<PathBuf>) -> Result<Option<Session>> {
    if !enabled {
        return Ok(None);
    }
    Session::new(output).map(Some)
}

pub fn print_summary(report: &Report) {
    if report.events.is_empty() {
        info!("benchmark: no events captured");
        return;
    }
    if let Some(path) = &report.output_path {
        info!(
            "benchmark: wrote {} events to {}",
            report.events.len(),
            path.display()
        );
    } else {
        info!("benchmark: captured {} events", report.events.len());
    }
    let mut totals = report.stage_totals.clone();
    totals.sort_by(|a, b| b.total.cmp(&a.total));
    for (idx, entry) in totals.iter().take(10).enumerate() {
        let total_ms = entry.total.as_secs_f64() * 1e3;
        let avg_ms = total_ms / entry.count as f64;
        let min_ms = entry.min.as_secs_f64() * 1e3;
        let max_ms = entry.max.as_secs_f64() * 1e3;
        let stage = &entry.stage;
        let count = entry.count;
        info!(
            "benchmark[{idx}]: stage={stage} count={count} total={total_ms:.3}ms avg={avg_ms:.3}ms min={min_ms:.3}ms max={max_ms:.3}ms"
        );
    }
}

#[cfg(feature = "bench-metrics")]
fn totals_for_events(events: &[BenchEvent]) -> Vec<StageTotal> {
    #[derive(Default)]
    struct Accum {
        count: usize,
        total: Duration,
        min: Option<Duration>,
        max: Option<Duration>,
    }

    let mut map: HashMap<&'static str, Accum> = HashMap::new();
    for event in events {
        let entry = map.entry(event.stage).or_default();
        entry.count += 1;
        entry.total += event.duration;
        entry.min = Some(match entry.min {
            Some(existing) if existing <= event.duration => existing,
            _ => event.duration,
        });
        entry.max = Some(match entry.max {
            Some(existing) if existing >= event.duration => existing,
            _ => event.duration,
        });
    }

    map.into_iter()
        .map(|(stage, accum)| StageTotal {
            stage: stage.to_owned(),
            count: accum.count,
            total: accum.total,
            min: accum.min.unwrap_or_default(),
            max: accum.max.unwrap_or_default(),
        })
        .collect()
}

#[cfg(feature = "bench-metrics")]
#[derive(Default)]
struct Collector {
    events: Mutex<Vec<BenchEvent>>,
}

#[cfg(feature = "bench-metrics")]
impl Collector {
    fn snapshot(&self) -> Vec<BenchEvent> {
        self.events
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }
}

#[cfg(feature = "bench-metrics")]
impl BenchRecorder for Collector {
    fn record(&self, event: BenchEvent) {
        if let Ok(mut guard) = self.events.lock() {
            guard.push(event);
        }
    }
}

#[cfg(feature = "bench-metrics")]
fn write_report(path: &PathBuf, events: &[BenchEvent]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create benchmark directory {}", parent.display())
            })?;
        }
    }
    let totals = totals_for_events(events);
    let value = json!({
        "events": events.iter().map(event_to_json).collect::<Vec<_>>(),
        "stage_totals": totals.iter().map(stage_to_json).collect::<Vec<_>>(),
    });
    let data = serde_json::to_vec_pretty(&value)?;
    fs::write(path, data)
        .with_context(|| format!("failed to write benchmark output {}", path.display()))?;
    Ok(())
}

#[cfg(feature = "bench-metrics")]
fn event_to_json(event: &BenchEvent) -> serde_json::Value {
    json!({
        "stage": event.stage,
        "duration_ms": event.duration.as_secs_f64() * 1e3,
        "duration_ns": event.duration.as_nanos().to_string(),
        "fields": event.fields.iter().map(|field| {
            json!({
                "key": field.key,
                "value": bench_value_to_json(&field.value),
            })
        }).collect::<Vec<_>>(),
    })
}

#[cfg(feature = "bench-metrics")]
fn stage_to_json(stage: &StageTotal) -> serde_json::Value {
    json!({
        "stage": stage.stage,
        "count": stage.count,
        "total_ms": stage.total.as_secs_f64() * 1e3,
        "total_ns": stage.total.as_nanos().to_string(),
        "avg_ms": if stage.count > 0 {
            stage.total.as_secs_f64() * 1e3 / stage.count as f64
        } else {
            0.0
        },
        "min_ms": stage.min.as_secs_f64() * 1e3,
        "max_ms": stage.max.as_secs_f64() * 1e3,
    })
}

#[cfg(feature = "bench-metrics")]
fn bench_value_to_json(value: &BenchValue) -> serde_json::Value {
    match value {
        BenchValue::U64(v) => json!(v),
        BenchValue::I64(v) => json!(v),
        BenchValue::F64(v) => json!(v),
        BenchValue::Bool(v) => json!(v),
        BenchValue::Text(v) => json!(v),
    }
}
