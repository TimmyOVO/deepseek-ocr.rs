use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    thread,
};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use deepseek_ocr_dsq::{DsqBiasDType, DsqHeader, DsqReader, DsqRecord, DsqTensorDType};
use deepseek_ocr_dsq_models::{
    AdapterRegistry, AdapterScope, LinearSpec, ModelAdapter, QuantContext,
};
use deepseek_ocr_dsq_writer::{
    encode_bias_values, quantize_q4k, quantize_q6k, quantize_q8_0, DsqWriter, SnapshotMetadata,
};
use half::{bf16, f16};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use memmap2::MmapOptions;
use rayon::ThreadPoolBuilder;
use rayon::{prelude::*, ThreadPool};
use safetensors::{tensor::TensorView, Dtype as SafeDType, SafeTensorError, SafeTensors};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};
use tracing_subscriber::EnvFilter;

const DEFAULT_CONFIG_PATH: &str = "DeepSeek-OCR/config.json";
const DEFAULT_WEIGHTS_PATH: &str = "DeepSeek-OCR/model-00001-of-000001.safetensors";

#[derive(Parser)]
#[command(
    name = "dsq",
    author,
    version,
    about = "Tools for inspecting DeepSeek-OCR DSQ snapshots"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Print header metadata and tensor records from a DSQ container.
    Info(InfoArgs),
    /// Print aggregate statistics for a DSQ container.
    Stats(StatsArgs),
    /// Export a quantized DSQ snapshot from safetensors weights.
    Export(ExportArgs),
}

#[derive(Parser)]
struct InfoArgs {
    /// Path to the `.dsq` file.
    path: PathBuf,
    /// Show every record (overrides --records).
    #[arg(long)]
    all: bool,
    /// Number of records to print when --all is not set.
    #[arg(
        short = 'n',
        long = "records",
        default_value_t = 5,
        conflicts_with = "all"
    )]
    records: usize,
}

#[derive(Parser)]
struct StatsArgs {
    /// Path to the `.dsq` file.
    path: PathBuf,
    /// Emit JSON summary to stdout (single line).
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum QuantDTypeArg {
    #[value(name = "Q8_0")]
    Q8_0,
    #[value(name = "Q4_K")]
    Q4K,
    #[value(name = "Q6_K")]
    Q6K,
}

impl QuantDTypeArg {
    fn label(self) -> &'static str {
        match self {
            Self::Q8_0 => "Q8_0",
            Self::Q4K => "Q4_K",
            Self::Q6K => "Q6_K",
        }
    }

    fn to_dtype(self) -> DsqTensorDType {
        match self {
            Self::Q8_0 => DsqTensorDType::Q8_0,
            Self::Q4K => DsqTensorDType::Q4K,
            Self::Q6K => DsqTensorDType::Q6K,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ExportTargets {
    #[value(name = "text")]
    Text,
    #[value(name = "text+projector")]
    TextAndProjector,
}

impl ExportTargets {
    fn to_scope(self) -> AdapterScope {
        match self {
            Self::Text => AdapterScope::Text,
            Self::TextAndProjector => AdapterScope::TextAndProjector,
        }
    }
}

#[derive(Parser)]
struct ExportArgs {
    /// Path to the DeepSeek OCR config JSON.
    #[arg(long, default_value = DEFAULT_CONFIG_PATH)]
    config: PathBuf,
    /// Path to the safetensors checkpoint shard.
    #[arg(long, default_value = DEFAULT_WEIGHTS_PATH)]
    weights: PathBuf,
    /// Output path for the generated `.dsq`.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Primary quantized dtype to use.
    #[arg(long, value_enum, default_value = "Q8_0")]
    dtype: QuantDTypeArg,
    /// Export scope (`text` or `text+projector`).
    #[arg(long, value_enum, default_value = "text")]
    targets: ExportTargets,
    /// Adapter identifier to force (defaults to auto-detect).
    #[arg(long)]
    adapter: Option<String>,
    /// Backend label recorded in the snapshot header.
    #[arg(long, default_value = "CPU")]
    backend: String,
    /// Candle version recorded in the snapshot metadata.
    #[arg(long, default_value = "unknown")]
    candle_version: String,
    /// Optional model identifier (defaults to SHA256(weights)).
    #[arg(long)]
    model_id: Option<String>,
    /// Allow skipping tensors that fail alignment checks.
    #[arg(long)]
    allow_skip: bool,
    /// Maximum parallel jobs (not yet implemented).
    #[arg(long)]
    jobs: Option<usize>,
}

fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Commands::Info(args) => run_info(args),
        Commands::Stats(args) => run_stats(args),
        Commands::Export(args) => run_export(args),
    }
}

fn run_info(args: InfoArgs) -> Result<()> {
    if !args.all && args.records == 0 {
        bail!("--records must be greater than zero");
    }
    let reader = DsqReader::open(&args.path)
        .with_context(|| format!("failed to open snapshot {}", args.path.display()))?;
    info!(
        artifact = %args.path.display(),
        tensors = reader.records().len(),
        "opened snapshot"
    );
    log_header(reader.header());
    let total = reader.records().len();
    if total == 0 {
        info!("no tensor records found");
        return Ok(());
    }
    let limit = if args.all {
        total
    } else {
        args.records.min(total)
    };
    info!(showing = limit, total, "listing tensor records");
    for (idx, record) in reader.records().iter().take(limit).enumerate() {
        log_record(idx, record);
    }
    if !args.all && limit < total {
        let remaining = total - limit;
        warn!(
            remaining,
            "additional record(s) omitted; rerun with --all to display everything"
        );
    }
    Ok(())
}

#[derive(Serialize)]
struct DtypeStats {
    count: usize,
    q_bytes: u64,
    bias_bytes: u64,
}

#[derive(Serialize)]
struct StatsSummary {
    tensor_total: usize,
    q_bytes_total: u64,
    bias_bytes_total: u64,
    dtypes: BTreeMap<String, DtypeStats>,
}

fn run_stats(args: StatsArgs) -> Result<()> {
    let reader = DsqReader::open(&args.path)
        .with_context(|| format!("failed to open snapshot {}", args.path.display()))?;
    let mut q_total: u64 = 0;
    let mut bias_total: u64 = 0;
    let mut by_dtype: BTreeMap<String, DtypeStats> = BTreeMap::new();
    for rec in reader.records() {
        q_total = q_total.saturating_add(rec.q_len);
        let b = rec.bias_len.unwrap_or(0);
        bias_total = bias_total.saturating_add(b);
        let key = rec.q_dtype.to_string();
        let entry = by_dtype.entry(key).or_insert(DtypeStats {
            count: 0,
            q_bytes: 0,
            bias_bytes: 0,
        });
        entry.count += 1;
        entry.q_bytes = entry.q_bytes.saturating_add(rec.q_len);
        entry.bias_bytes = entry.bias_bytes.saturating_add(b);
    }
    let summary = StatsSummary {
        tensor_total: reader.records().len(),
        q_bytes_total: q_total,
        bias_bytes_total: bias_total,
        dtypes: by_dtype,
    };
    if args.json {
        println!("{}", serde_json::to_string(&summary)?);
    } else {
        info!(
            tensors = summary.tensor_total,
            q_bytes = summary.q_bytes_total,
            bias_bytes = summary.bias_bytes_total,
            unique_dtypes = summary.dtypes.len(),
            "snapshot stats"
        );
        for (dtype, st) in &summary.dtypes {
            let pct = if summary.tensor_total > 0 {
                (st.count as f64) * 100.0 / (summary.tensor_total as f64)
            } else {
                0.0
            };
            info!(
                dtype = %dtype,
                count = st.count,
                percent = format!("{pct:.2}"),
                q_bytes = st.q_bytes,
                bias_bytes = st.bias_bytes,
                "dtype stats"
            );
        }
    }
    Ok(())
}

fn run_export(args: ExportArgs) -> Result<()> {
    let primary_dtype = args.dtype.to_dtype();
    let config = load_config_value(&args.config).with_context(|| {
        format!(
            "failed to parse model config file {}",
            args.config.display()
        )
    })?;
    let scope = args.targets.to_scope();
    let registry = AdapterRegistry::global();
    let adapter = match args.adapter.as_deref() {
        Some(id) => registry.get(id).with_context(|| {
            format!(
                "unknown adapter `{id}`; available adapters: {}",
                format_adapter_list(registry)
            )
        })?,
        None => registry.infer_adapter(&config).with_context(|| {
            format!(
                "failed to infer model adapter from config {}; pass --adapter to select explicitly (available: {})",
                args.config.display(),
                format_adapter_list(registry)
            )
        })?,
    };
    let specs = adapter.discover(&config, scope).with_context(|| {
        format!(
            "failed to derive tensor layout from config {} with adapter {}",
            args.config.display(),
            adapter.id()
        )
    })?;
    if specs.is_empty() {
        bail!(
            "no linear tensors discovered from config {}; cannot export snapshot",
            args.config.display()
        );
    }
    let model_id = match args.model_id {
        Some(id) => id,
        None => compute_sha256_hex(&args.weights).with_context(|| {
            format!(
                "failed to compute SHA256 digest for weights {}",
                args.weights.display()
            )
        })?,
    };
    let default_output = PathBuf::from(format!("{}.{}.dsq", model_id, args.dtype.label()));
    let mut artifact_path = args.output.unwrap_or(default_output);
    artifact_path.set_extension("dsq");
    if let Some(parent) = artifact_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
    }
    info!(
        artifact = %artifact_path.display(),
        tensors = specs.len(),
        dtype = %primary_dtype,
        adapter = adapter.id(),
        auto_fallback = true,
        include_projector = scope.includes_projector(),
        "starting DSQ export"
    );
    let weights_file = File::open(&args.weights)
        .with_context(|| format!("failed to open weights {}", args.weights.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&weights_file) }
        .with_context(|| format!("failed to mmap {}", args.weights.display()))?;
    let tensors = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("failed to parse safetensors {}", args.weights.display()))?;
    let metadata = SnapshotMetadata {
        candle_version: args.candle_version.clone(),
        model_id: model_id.clone(),
        backend: args.backend.clone(),
        default_qdtype: primary_dtype,
    };
    let mut writer = DsqWriter::new(&artifact_path, metadata).with_context(|| {
        format!(
            "failed to initialize DSQ writer at {}",
            artifact_path.display()
        )
    })?;
    let pool = match args.jobs {
        Some(threads) if threads > 0 => Some(
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .context("failed to initialize rayon thread pool")?,
        ),
        _ => None,
    };
    let chunk_size = determine_chunk_size(specs.len(), args.jobs);
    let progress = make_progress_bar(specs.len());
    let quant_ctx = QuantContext {
        primary: primary_dtype,
    };
    let stats = export_tensors(
        &mut writer,
        &tensors,
        &specs,
        primary_dtype,
        args.allow_skip,
        progress.as_ref(),
        pool.as_ref(),
        chunk_size,
        adapter,
        &quant_ctx,
    )?;
    if let Some(pb) = &progress {
        pb.finish_and_clear();
    }
    writer
        .finalize()
        .with_context(|| format!("failed to finalize snapshot at {}", artifact_path.display()))?;
    info!(
        artifact = %artifact_path.display(),
        tensors = stats.exported,
        skipped = stats.skipped,
        fallbacks = stats.fallbacks,
        q_bytes = stats.q_bytes,
        bias_bytes = stats.bias_bytes,
        adapter = adapter.id(),
        dtype_breakdown = %format_dtype_breakdown(&stats.dtype_counts),
        "snapshot export complete"
    );
    Ok(())
}

fn make_progress_bar(total: usize) -> Option<ProgressBar> {
    if total == 0 {
        return None;
    }
    let pb = ProgressBar::new(total as u64);
    pb.set_draw_target(ProgressDrawTarget::stderr());
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} Quantizing {pos}/{len} tensors [{elapsed_precise}]",
        )
        .unwrap()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ "),
    );
    Some(pb)
}

fn format_adapter_list(registry: &AdapterRegistry) -> String {
    registry
        .list()
        .iter()
        .map(|adapter| (*adapter).id())
        .collect::<Vec<_>>()
        .join(", ")
}

fn determine_chunk_size(total_specs: usize, jobs: Option<usize>) -> usize {
    if total_specs == 0 {
        return 1;
    }
    let threads = jobs
        .filter(|&j| j > 0)
        .or_else(|| thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1);
    let chunk = threads.saturating_mul(4);
    chunk.clamp(1, total_specs)
}

#[derive(Default)]
struct ExportStats {
    exported: usize,
    skipped: usize,
    q_bytes: u64,
    bias_bytes: u64,
    fallbacks: usize,
    dtype_counts: BTreeMap<String, DTypeBreakdown>,
}

#[derive(Default)]
struct DTypeBreakdown {
    tensors: usize,
    q_bytes: u64,
}

impl ExportStats {
    fn record(
        &mut self,
        dtype: DsqTensorDType,
        q_bytes: u64,
        bias_bytes: u64,
        fallback_used: bool,
    ) {
        self.exported += 1;
        self.q_bytes = self.q_bytes.saturating_add(q_bytes);
        self.bias_bytes = self.bias_bytes.saturating_add(bias_bytes);
        if fallback_used {
            self.fallbacks += 1;
        }
        let entry = self.dtype_counts.entry(dtype.to_string()).or_default();
        entry.tensors += 1;
        entry.q_bytes = entry.q_bytes.saturating_add(q_bytes);
    }
}

fn format_dtype_breakdown(map: &BTreeMap<String, DTypeBreakdown>) -> String {
    if map.is_empty() {
        return "none".to_string();
    }
    map.iter()
        .map(|(dtype, stats)| {
            format!(
                "{}: {} tensor(s), {} byte(s)",
                dtype, stats.tensors, stats.q_bytes
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

enum QuantTaskResult {
    Quantized {
        name: String,
        out_dim: usize,
        in_dim: usize,
        dtype: DsqTensorDType,
        qbytes: Vec<u8>,
        bias: Option<Vec<f32>>,
        fallback_from: Option<DsqTensorDType>,
    },
    Float {
        name: String,
        out_dim: usize,
        in_dim: usize,
        payload: FloatPayload,
        bias: Option<Vec<f32>>,
    },
    #[allow(dead_code)]
    Skipped,
}

enum FloatPayload {
    F32(Vec<f32>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
}

impl FloatPayload {
    fn dtype(&self) -> DsqTensorDType {
        match self {
            Self::F32(_) => DsqTensorDType::F32,
            Self::F16(_) => DsqTensorDType::F16,
            Self::BF16(_) => DsqTensorDType::BF16,
        }
    }

    fn byte_len(&self) -> u64 {
        match self {
            Self::F32(values) => (values.len() * std::mem::size_of::<f32>()) as u64,
            Self::F16(values) => (values.len() * std::mem::size_of::<f16>()) as u64,
            Self::BF16(values) => (values.len() * std::mem::size_of::<bf16>()) as u64,
        }
    }
}

fn process_chunk(
    chunk: &[&LinearSpec],
    tensors: &SafeTensors<'_>,
    primary: DsqTensorDType,
    allow_skip: bool,
    progress: Option<&ProgressBar>,
    adapter: &dyn ModelAdapter,
    ctx: &QuantContext,
) -> Result<Vec<QuantTaskResult>> {
    chunk
        .par_iter()
        .map(|spec| quantize_spec(spec, tensors, primary, allow_skip, progress, adapter, ctx))
        .collect()
}

fn quantize_spec(
    spec: &LinearSpec,
    tensors: &SafeTensors<'_>,
    primary: DsqTensorDType,
    _allow_skip: bool,
    progress: Option<&ProgressBar>,
    adapter: &dyn ModelAdapter,
    ctx: &QuantContext,
) -> Result<QuantTaskResult> {
    let tensor = tensors.tensor(&spec.name).map_err(|err| match err {
        SafeTensorError::TensorNotFound(_) => {
            anyhow::anyhow!("tensor `{}` not found in checkpoint", spec.name)
        }
        other => other.into(),
    })?;
    validate_weight_shape(tensor.shape(), spec.out_dim, spec.in_dim)?;
    let weights = tensor_to_f32(&tensor)?;
    let expected_len = spec
        .out_dim
        .checked_mul(spec.in_dim)
        .context("tensor element count overflow")?;
    if weights.len() != expected_len {
        bail!(
            "tensor `{}` provides {} values, expected {}",
            spec.name,
            weights.len(),
            expected_len
        );
    }
    let bias = match spec.bias.as_deref() {
        Some(name) => match tensors.tensor(name) {
            Ok(view) => {
                validate_bias_shape(view.shape(), spec.out_dim)?;
                let values = tensor_to_f32(&view)?;
                Some(values)
            }
            Err(SafeTensorError::TensorNotFound(_)) => None,
            Err(err) => return Err(err.into()),
        },
        None => None,
    };
    let requested_dtype = adapter
        .recommend_dtype(&spec.name, spec.in_dim, ctx)
        .unwrap_or(primary);
    match select_dtype(requested_dtype, spec.in_dim) {
        Ok(selection) => {
            if let Some(prev) = selection.fallback_from {
                let reason = selection.reason.as_deref().unwrap_or("alignment mismatch");
                warn!(
                    tensor = %spec.name,
                    requested = %prev,
                    selected = %selection.dtype,
                    reason = %reason,
                    "auto fallback applied"
                );
            }
            let qbytes = match selection.dtype {
                DsqTensorDType::Q8_0 => quantize_q8_0(&weights, spec.out_dim, spec.in_dim)?,
                DsqTensorDType::Q4K => quantize_q4k(&weights, spec.out_dim, spec.in_dim)?,
                DsqTensorDType::Q6K => quantize_q6k(&weights, spec.out_dim, spec.in_dim)?,
                other => unreachable!("float dtype {other:?} cannot be selected here"),
            };
            if let Some(pb) = progress {
                pb.inc(1);
            }
            Ok(QuantTaskResult::Quantized {
                name: spec.name.clone(),
                out_dim: spec.out_dim,
                in_dim: spec.in_dim,
                dtype: selection.dtype,
                qbytes,
                bias,
                fallback_from: selection.fallback_from,
            })
        }
        Err(err) => {
            let float_dtype = select_float_dtype(tensor.dtype());
            warn!(
                tensor = %spec.name,
                requested = %requested_dtype,
                selected = %float_dtype,
                reason = %err.message,
                "falling back to float tensor"
            );
            let payload = build_float_payload(
                &tensor,
                float_dtype,
                spec.out_dim,
                spec.in_dim,
                weights,
                &spec.name,
            )?;
            if let Some(pb) = progress {
                pb.inc(1);
            }
            Ok(QuantTaskResult::Float {
                name: spec.name.clone(),
                out_dim: spec.out_dim,
                in_dim: spec.in_dim,
                payload,
                bias,
            })
        }
    }
}

fn load_config_value(path: &Path) -> Result<Value> {
    let file =
        File::open(path).with_context(|| format!("failed to open config {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("failed to parse JSON {}", path.display()))
}

fn export_tensors(
    writer: &mut DsqWriter,
    tensors: &SafeTensors<'_>,
    specs: &[LinearSpec],
    primary: DsqTensorDType,
    allow_skip: bool,
    progress: Option<&ProgressBar>,
    pool: Option<&ThreadPool>,
    chunk_size: usize,
    adapter: &dyn ModelAdapter,
    ctx: &QuantContext,
) -> Result<ExportStats> {
    let mut stats = ExportStats::default();
    let spec_refs: Vec<&LinearSpec> = specs.iter().collect();
    for chunk in spec_refs.chunks(chunk_size.max(1)) {
        let results = if let Some(pool) = pool {
            pool.install(|| {
                process_chunk(chunk, tensors, primary, allow_skip, progress, adapter, ctx)
            })
        } else {
            process_chunk(chunk, tensors, primary, allow_skip, progress, adapter, ctx)
        }?;
        for result in results {
            match result {
                QuantTaskResult::Quantized {
                    name,
                    out_dim,
                    in_dim,
                    dtype,
                    qbytes,
                    bias,
                    fallback_from,
                } => {
                    let bias_bytes = bias.as_ref().map(|values| encode_bias_values(values));
                    let bias_payload = bias_bytes
                        .as_ref()
                        .map(|bytes| (bytes.as_slice(), DsqBiasDType::F32));
                    writer.add_quantized_bytes(
                        &name,
                        out_dim,
                        in_dim,
                        dtype,
                        &qbytes,
                        bias_payload,
                    )?;
                    let q_len = qbytes.len() as u64;
                    let bias_len = bias_bytes.as_ref().map(|b| b.len() as u64).unwrap_or(0);
                    stats.record(dtype, q_len, bias_len, fallback_from.is_some());
                    debug!(
                        tensor = %name,
                        dtype = %dtype,
                        q_bytes = q_len,
                        bias_bytes = bias_len,
                        "quantized tensor"
                    );
                }
                QuantTaskResult::Float {
                    name,
                    out_dim,
                    in_dim,
                    payload,
                    bias,
                } => {
                    let dtype = payload.dtype();
                    let q_len = payload.byte_len();
                    let bias_len = bias
                        .as_ref()
                        .map(|vals| (vals.len() * std::mem::size_of::<f32>()) as u64)
                        .unwrap_or(0);
                    let bias_slice = bias.as_deref();
                    match payload {
                        FloatPayload::F32(values) => {
                            writer.add_f32_tensor(&name, out_dim, in_dim, &values, bias_slice)?
                        }
                        FloatPayload::F16(values) => {
                            writer.add_f16_tensor(&name, out_dim, in_dim, &values, bias_slice)?
                        }
                        FloatPayload::BF16(values) => {
                            writer.add_bf16_tensor(&name, out_dim, in_dim, &values, bias_slice)?
                        }
                    }
                    stats.record(dtype, q_len, bias_len, true);
                    debug!(
                        tensor = %name,
                        dtype = %dtype,
                        q_bytes = q_len,
                        bias_bytes = bias_len,
                        "wrote float tensor"
                    );
                }
                QuantTaskResult::Skipped => {
                    stats.skipped += 1;
                }
            }
        }
    }
    Ok(stats)
}

struct SelectionResult {
    dtype: DsqTensorDType,
    fallback_from: Option<DsqTensorDType>,
    reason: Option<String>,
}

struct SelectionError {
    message: String,
}

fn select_dtype(primary: DsqTensorDType, in_dim: usize) -> Result<SelectionResult, SelectionError> {
    let mut current = primary;
    let mut fallback_from = None;
    let mut reason = None;
    let mut attempts = Vec::new();

    loop {
        let block = dtype_block_size(current);
        if in_dim % block == 0 {
            return Ok(SelectionResult {
                dtype: current,
                fallback_from,
                reason,
            });
        }
        let failure = format!(
            "input dim {} not divisible by {} ({})",
            in_dim, block, current
        );
        attempts.push((current, block));
        if let Some(next) = next_fallback_dtype(current) {
            if fallback_from.is_none() {
                fallback_from = Some(current);
                reason = Some(failure.clone());
            }
            current = next;
            continue;
        }
        let attempt_chain = attempts
            .into_iter()
            .map(|(dtype, block)| format!("{} (block {block})", dtype))
            .collect::<Vec<_>>()
            .join(" -> ");
        return Err(SelectionError {
            message: format!("{failure}; attempted chain: {attempt_chain}"),
        });
    }
}

fn next_fallback_dtype(dtype: DsqTensorDType) -> Option<DsqTensorDType> {
    match dtype {
        DsqTensorDType::Q6K | DsqTensorDType::Q4K => Some(DsqTensorDType::Q8_0),
        DsqTensorDType::Q8_0 => None,
        _ => None,
    }
}

fn dtype_block_size(dtype: DsqTensorDType) -> usize {
    dtype
        .block_size()
        .expect("supported dtype must define block size")
}

fn select_float_dtype(dtype: SafeDType) -> DsqTensorDType {
    match dtype {
        SafeDType::BF16 => DsqTensorDType::BF16,
        SafeDType::F16 => DsqTensorDType::F16,
        _ => DsqTensorDType::F32,
    }
}

fn build_float_payload(
    view: &TensorView<'_>,
    dtype: DsqTensorDType,
    out_dim: usize,
    in_dim: usize,
    weights: Vec<f32>,
    name: &str,
) -> Result<FloatPayload> {
    let elements = out_dim
        .checked_mul(in_dim)
        .context("tensor element count overflow")?;
    match dtype {
        DsqTensorDType::F32 => Ok(FloatPayload::F32(weights)),
        DsqTensorDType::F16 => {
            drop(weights);
            let values = decode_f16_values(view, elements, name)?;
            Ok(FloatPayload::F16(values))
        }
        DsqTensorDType::BF16 => {
            drop(weights);
            let values = decode_bf16_values(view, elements, name)?;
            Ok(FloatPayload::BF16(values))
        }
        other => bail!("dtype {:?} is not supported for float fallback", other),
    }
}

fn decode_f16_values(view: &TensorView<'_>, elements: usize, name: &str) -> Result<Vec<f16>> {
    let data = view.data();
    let expected = elements
        .checked_mul(std::mem::size_of::<f16>())
        .context("f16 byte length overflow")?;
    if data.len() != expected {
        bail!(
            "tensor `{name}` f16 payload has {} bytes, expected {}",
            data.len(),
            expected
        );
    }
    let mut values = Vec::with_capacity(elements);
    for chunk in data.chunks_exact(2) {
        let bits = u16::from_le_bytes(chunk.try_into().expect("chunk len 2"));
        values.push(f16::from_bits(bits));
    }
    Ok(values)
}

fn decode_bf16_values(view: &TensorView<'_>, elements: usize, name: &str) -> Result<Vec<bf16>> {
    let data = view.data();
    let expected = elements
        .checked_mul(std::mem::size_of::<bf16>())
        .context("bf16 byte length overflow")?;
    if data.len() != expected {
        bail!(
            "tensor `{name}` bf16 payload has {} bytes, expected {}",
            data.len(),
            expected
        );
    }
    let mut values = Vec::with_capacity(elements);
    for chunk in data.chunks_exact(2) {
        let bits = u16::from_le_bytes(chunk.try_into().expect("chunk len 2"));
        values.push(bf16::from_bits(bits));
    }
    Ok(values)
}

fn tensor_to_f32(view: &TensorView<'_>) -> Result<Vec<f32>> {
    let data = view.data();
    let mut values = Vec::with_capacity(view.shape().iter().product());
    match view.dtype() {
        SafeDType::F32 => {
            for chunk in data.chunks_exact(4) {
                values.push(f32::from_le_bytes(chunk.try_into().expect("chunk len 4")));
            }
        }
        SafeDType::F16 => {
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().expect("chunk len 2"));
                values.push(f16::from_bits(bits).to_f32());
            }
        }
        SafeDType::BF16 => {
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().expect("chunk len 2"));
                values.push(bf16::from_bits(bits).to_f32());
            }
        }
        SafeDType::F64 => {
            for chunk in data.chunks_exact(8) {
                values.push(f64::from_le_bytes(chunk.try_into().expect("chunk len 8")) as f32);
            }
        }
        other => {
            bail!("tensor dtype {:?} is not supported for export", other);
        }
    }
    Ok(values)
}

fn validate_weight_shape(shape: &[usize], out_dim: usize, in_dim: usize) -> Result<()> {
    if shape.len() == 2 {
        if shape[0] != out_dim || shape[1] != in_dim {
            bail!(
                "tensor shape {} does not match expected {}x{}",
                format_shape(shape),
                out_dim,
                in_dim
            );
        }
        return Ok(());
    }
    let total: usize = shape.iter().product();
    let expected = out_dim
        .checked_mul(in_dim)
        .context("tensor element count overflow")?;
    if total != expected {
        bail!(
            "tensor shape {} encodes {} values, expected {}",
            format_shape(shape),
            total,
            expected
        );
    }
    Ok(())
}

fn validate_bias_shape(shape: &[usize], out_dim: usize) -> Result<()> {
    let total: usize = shape.iter().product();
    if total != out_dim {
        bail!(
            "bias shape {} encodes {} values, expected {}",
            format_shape(shape),
            total,
            out_dim
        );
    }
    Ok(())
}

fn format_shape(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(|dim| dim.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

fn compute_sha256_hex(path: &Path) -> Result<String> {
    let file = File::open(path)
        .with_context(|| format!("failed to open {} for hashing", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 20];
    loop {
        let read = reader
            .read(&mut buf)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn log_header(header: &DsqHeader) {
    info!(
        version = header.version,
        candle_version = %header.candle_version,
        model_id = %header.model_id,
        backend = %header.backend,
        default_qdtype = %header.default_qdtype,
        block_size = header.block_size,
        tensor_count = header.tensor_count,
        "snapshot header"
    );
}

fn log_record(idx: usize, record: &DsqRecord) {
    let bias = format_bias(record);
    info!(
        index = idx + 1,
        name = %record.name,
        out_dim = record.out_dim,
        in_dim = record.in_dim,
        q_dtype = %record.q_dtype,
        q_bytes = record.q_len,
        bias = %bias,
        "tensor record"
    );
}

fn format_bias(record: &DsqRecord) -> String {
    match (record.bias_dtype, record.bias_len) {
        (Some(dtype), Some(len)) => format!("{dtype} ({} bytes)", len),
        (None, None) => "none".to_string(),
        _ => "invalid metadata".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepseek_ocr_dsq::{DsqBiasDType, DsqTensorDType};

    #[test]
    fn bias_formatter_handles_all_cases() {
        let base = DsqRecord {
            name: "layer".to_string(),
            out_dim: 1,
            in_dim: 1,
            q_dtype: DsqTensorDType::Q8_0,
            q_offset: 0,
            q_len: 10,
            bias_offset: None,
            bias_len: None,
            bias_dtype: None,
        };
        assert_eq!(format_bias(&base), "none");

        let mut with_bias = base.clone();
        with_bias.bias_dtype = Some(DsqBiasDType::F32);
        with_bias.bias_len = Some(128);
        assert_eq!(format_bias(&with_bias), "F32 (128 bytes)");

        let mut invalid = base;
        invalid.bias_dtype = Some(DsqBiasDType::F16);
        assert_eq!(format_bias(&invalid), "invalid metadata");
    }

    #[test]
    fn quant_dtype_arg_maps_variants() {
        assert_eq!(QuantDTypeArg::Q8_0.to_dtype(), DsqTensorDType::Q8_0);
        assert_eq!(QuantDTypeArg::Q4K.to_dtype(), DsqTensorDType::Q4K);
        assert_eq!(QuantDTypeArg::Q6K.to_dtype(), DsqTensorDType::Q6K);
    }
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}
