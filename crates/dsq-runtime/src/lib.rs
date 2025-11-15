use std::{
    collections::HashMap,
    env,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, ensure, Context, Result};
use candle_core::{
    quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul},
    Device, Tensor,
};
use deepseek_ocr_dsq::{DsqBiasDType, DsqHeader, DsqReader, DsqRecord, DsqTensorDType};
use half::{bf16, f16};
use rayon::{current_thread_index, prelude::*, ThreadPoolBuilder};
use tracing::{debug, info};

/// Location of the design document describing the snapshot contract.
pub const SNAPSHOT_SPEC_PATH: &str = "docs/quant_snapshot.md";

const SNAPSHOT_DSQ_MAGIC: &[u8; 7] = b"DSQSNAP";

#[derive(Debug, Clone)]
enum SnapshotStorage {
    Dsq { path: PathBuf },
}

impl SnapshotStorage {
    fn label(&self) -> &'static str {
        "dsq"
    }

    fn path(&self) -> &Path {
        match self {
            SnapshotStorage::Dsq { path } => path.as_path(),
        }
    }
}

/// Concrete snapshot instance backed by a parsed DSQ reader.
pub struct QuantizedSnapshot {
    reader: DsqReader,
    storage: SnapshotStorage,
    runtime_cfg: SnapshotRuntimeConfig,
}

/// Return value when reconstructing a linear layer from the snapshot.
#[derive(Debug)]
pub enum SnapshotLinear {
    /// Quantized tensor reconstructed as a `QMatMul`.
    Quantized {
        qmatmul: Arc<QMatMul>,
        bias: Option<Tensor>,
    },
    /// Float tensor decoded into a dense `Tensor`.
    Float {
        weight: Tensor,
        bias: Option<Tensor>,
    },
}

/// Lookup map from tensor name to reconstructed payload.
pub type SnapshotLinearMap = HashMap<String, SnapshotLinear>;

/// Specification describing a linear tensor we expect to find in the snapshot.
#[derive(Debug, Clone)]
pub struct LinearSpec {
    pub name: String,
    pub out_dim: usize,
    pub in_dim: usize,
}

impl LinearSpec {
    pub fn new(name: impl Into<String>, out_dim: usize, in_dim: usize) -> Self {
        Self {
            name: name.into(),
            out_dim,
            in_dim,
        }
    }

    pub fn elements(&self) -> Option<usize> {
        self.out_dim.checked_mul(self.in_dim)
    }
}

/// Batch of tensors that should be reconstructed together.
#[derive(Debug, Clone, Default)]
pub struct SnapshotLoadPlan {
    linears: Vec<LinearSpec>,
}

impl SnapshotLoadPlan {
    pub fn new(linears: Vec<LinearSpec>) -> Self {
        Self { linears }
    }

    pub fn push(&mut self, spec: LinearSpec) {
        self.linears.push(spec);
    }

    pub fn extend<I: IntoIterator<Item = LinearSpec>>(&mut self, specs: I) {
        self.linears.extend(specs);
    }

    pub fn linears(&self) -> &[LinearSpec] {
        &self.linears
    }

    pub fn len(&self) -> usize {
        self.linears.len()
    }

    pub fn is_empty(&self) -> bool {
        self.linears.is_empty()
    }

    pub fn execute(
        self,
        snapshot: Option<&QuantizedSnapshot>,
        device: &Device,
        parallel: Option<&ParallelConfig>,
    ) -> Result<Option<SnapshotLinearMap>> {
        match snapshot {
            Some(snapshot) if !self.is_empty() => {
                let cfg = parallel.unwrap_or_else(|| snapshot.runtime_config().parallel());
                snapshot.load_linear_map(&self, device, cfg).map(Some)
            }
            _ => Ok(None),
        }
    }
}

/// Result of executing a [`SnapshotLoadPlan`]. Order matches the requested plan.
#[derive(Debug)]
pub struct LoadedLinear {
    pub spec: LinearSpec,
    pub payload: Option<SnapshotLinear>,
}

impl LoadedLinear {
    pub fn new(spec: LinearSpec, payload: Option<SnapshotLinear>) -> Self {
        Self { spec, payload }
    }

    pub fn name(&self) -> &str {
        &self.spec.name
    }

    pub fn payload(&self) -> Option<&SnapshotLinear> {
        self.payload.as_ref()
    }
}

/// Controls how aggressively snapshot linears should be loaded in parallel.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    pub enabled: bool,
    pub max_jobs: usize,
    pub min_tensor_elements: usize,
}

impl ParallelConfig {
    pub const fn serial() -> Self {
        Self {
            enabled: false,
            max_jobs: 1,
            min_tensor_elements: 0,
        }
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self::serial()
    }
}

/// Snapshot runtime configuration sourced from environment variables.
#[derive(Debug, Clone)]
pub struct SnapshotRuntimeConfig {
    parallel: ParallelConfig,
}

impl SnapshotRuntimeConfig {
    pub fn from_env() -> Self {
        const MIN_ELEMS_DEFAULT: usize = 1 << 14;
        let enable_parallel = env_bool("DEEPSEEK_SNAPSHOT_PARALLEL").unwrap_or(false);
        let max_jobs = env::var("DEEPSEEK_SNAPSHOT_MAX_JOBS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .or_else(|| std::thread::available_parallelism().ok().map(|v| v.get()))
            .unwrap_or(1);
        let min_tensor_elements = env::var("DEEPSEEK_SNAPSHOT_MIN_TENSOR_ELEMENTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(MIN_ELEMS_DEFAULT);
        let parallel = if enable_parallel && max_jobs > 1 {
            ParallelConfig {
                enabled: true,
                max_jobs,
                min_tensor_elements,
            }
        } else {
            ParallelConfig::serial()
        };
        Self { parallel }
    }

    pub const fn serial() -> Self {
        Self {
            parallel: ParallelConfig::serial(),
        }
    }

    pub fn parallel(&self) -> &ParallelConfig {
        &self.parallel
    }
}

impl Default for SnapshotRuntimeConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

/// Return `true` when the build claims to support direct QTensor byte IO.
pub const fn qtensor_bytes_supported() -> bool {
    cfg!(feature = "qtensor-bytes")
}

impl QuantizedSnapshot {
    /// Load a snapshot from a `.dsq` file or a directory containing one.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        match resolve_snapshot_storage(path.as_ref())? {
            storage @ SnapshotStorage::Dsq { .. } => Self::load_dsq(storage),
        }
    }

    pub fn header(&self) -> &DsqHeader {
        self.reader.header()
    }

    pub fn tensor_record(&self, name: &str) -> Option<&DsqRecord> {
        self.reader.tensor(name)
    }

    pub fn tensor_records(&self) -> &[DsqRecord] {
        self.reader.records()
    }

    pub fn container_label(&self) -> &'static str {
        self.storage.label()
    }

    pub fn runtime_config(&self) -> &SnapshotRuntimeConfig {
        &self.runtime_cfg
    }

    /// Load a batch of linears according to the provided plan.
    pub fn load_linears(
        &self,
        plan: &SnapshotLoadPlan,
        device: &Device,
        parallel: &ParallelConfig,
    ) -> Result<Vec<LoadedLinear>> {
        if !should_parallelize(plan, parallel) {
            return self.load_linears_serial(plan, device, parallel);
        }
        let num_threads = parallel.max_jobs.max(1);
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .with_context(|| {
                format!(
                    "failed to build snapshot parallel pool with {} threads",
                    num_threads
                )
            })?;
        debug!(
            tensors = plan.len(),
            threads = num_threads,
            "snapshot parallel load"
        );
        pool.install(|| self.load_linears_parallel(plan, device, parallel))
    }

    pub fn load_linear_map(
        &self,
        plan: &SnapshotLoadPlan,
        device: &Device,
        parallel: &ParallelConfig,
    ) -> Result<SnapshotLinearMap> {
        let loaded = self.load_linears(plan, device, parallel)?;
        Ok(loaded
            .into_iter()
            .filter_map(|linear| linear.payload.map(|payload| (linear.spec.name, payload)))
            .collect())
    }

    /// Attempt to construct a quantized linear layer for the given tensor.
    pub fn load_linear(
        &self,
        name: &str,
        out_dim: usize,
        in_dim: usize,
        device: &Device,
    ) -> Result<Option<SnapshotLinear>> {
        let spec = LinearSpec::new(name, out_dim, in_dim);
        self.load_linear_from_spec(&spec, device, self.runtime_cfg.parallel())
    }

    fn load_linear_from_spec(
        &self,
        spec: &LinearSpec,
        device: &Device,
        parallel: &ParallelConfig,
    ) -> Result<Option<SnapshotLinear>> {
        let name = &spec.name;
        let record = match self.reader.tensor(name) {
            Some(record) => record,
            None => return Ok(None),
        };
        if record.out_dim != spec.out_dim || record.in_dim != spec.in_dim {
            bail!(
                "snapshot tensor `{name}` dims mismatch (expected {}x{}, found {}x{})",
                spec.out_dim,
                spec.in_dim,
                record.out_dim,
                record.in_dim
            );
        }
        let qweight_bytes = self.reader.tensor_bytes(record)?;
        let bias = self.load_bias(record, name, device)?;
        match record.q_dtype {
            DsqTensorDType::Q8_0 | DsqTensorDType::Q4K | DsqTensorDType::Q6K => {
                let ggml_dtype = ggml_from_snapshot_dtype(record.q_dtype)?;
                let qtensor = qtensor_from_ggml(
                    ggml_dtype,
                    qweight_bytes,
                    vec![record.out_dim, record.in_dim],
                    device,
                )
                .with_context(|| {
                    format!("failed to construct qtensor for `{name}` from snapshot")
                })?;
                let qmatmul = Arc::new(
                    QMatMul::from_qtensor(qtensor)
                        .with_context(|| format!("failed to build QMatMul for `{name}`"))?,
                );
                Ok(Some(SnapshotLinear::Quantized { qmatmul, bias }))
            }
            DsqTensorDType::F16 | DsqTensorDType::BF16 | DsqTensorDType::F32 => {
                let weight = self.dense_tensor(
                    name,
                    record.q_dtype,
                    qweight_bytes,
                    record.out_dim,
                    record.in_dim,
                    device,
                    parallel,
                )?;
                Ok(Some(SnapshotLinear::Float { weight, bias }))
            }
        }
    }

    fn load_linears_serial(
        &self,
        plan: &SnapshotLoadPlan,
        device: &Device,
        parallel: &ParallelConfig,
    ) -> Result<Vec<LoadedLinear>> {
        plan.linears()
            .iter()
            .map(|spec| {
                let payload = self.load_linear_from_spec(spec, device, parallel)?;
                Ok(LoadedLinear::new(spec.clone(), payload))
            })
            .collect()
    }

    fn load_linears_parallel(
        &self,
        plan: &SnapshotLoadPlan,
        device: &Device,
        parallel: &ParallelConfig,
    ) -> Result<Vec<LoadedLinear>> {
        plan.linears()
            .par_iter()
            .map(|spec| -> Result<LoadedLinear> {
                let payload = self.load_linear_from_spec(spec, device, parallel)?;
                Ok(LoadedLinear::new(spec.clone(), payload))
            })
            .collect::<Result<Vec<_>>>()
    }

    fn load_dsq(storage: SnapshotStorage) -> Result<Self> {
        let path = storage.path().to_path_buf();
        let reader = DsqReader::open(&path)
            .with_context(|| format!("failed to load snapshot at {}", path.display()))?;
        Self::from_reader(reader, storage)
    }

    fn from_reader(reader: DsqReader, storage: SnapshotStorage) -> Result<Self> {
        let header = reader.header();
        ensure_supported_snapshot_dtype(header.default_qdtype)?;
        for record in reader.records() {
            ensure_supported_snapshot_dtype(record.q_dtype)?;
        }
        let SnapshotStorage::Dsq { path } = &storage;
        info!(
            tensors = header.tensor_count,
            model_id = %header.model_id,
            candle_version = %header.candle_version,
            backend = %header.backend,
            container = storage.label(),
            artifact = %path.display(),
            "loaded {:?} snapshot",
            header.default_qdtype
        );
        Ok(Self {
            reader,
            storage,
            runtime_cfg: SnapshotRuntimeConfig::default(),
        })
    }

    fn load_bias(&self, record: &DsqRecord, name: &str, device: &Device) -> Result<Option<Tensor>> {
        if let Some(dtype) = record.bias_dtype {
            let bytes = self
                .reader
                .bias_bytes(record)?
                .context(format!("bias metadata missing for `{name}` despite dtype"))?;
            Ok(Some(self.bias_tensor(
                bytes,
                dtype,
                record.out_dim,
                device,
            )?))
        } else {
            Ok(None)
        }
    }

    fn dense_tensor(
        &self,
        name: &str,
        dtype: DsqTensorDType,
        bytes: &[u8],
        out_dim: usize,
        in_dim: usize,
        device: &Device,
        parallel: &ParallelConfig,
    ) -> Result<Tensor> {
        let elements = out_dim
            .checked_mul(in_dim)
            .context("linear element count overflow")?;
        let parallel_dense = should_parallelize_tensor(elements, parallel);
        if parallel_dense {
            debug!(tensor = name, dtype = ?dtype, elements, "snapshot parallel dense decode");
        }
        let tensor = match dtype {
            DsqTensorDType::F32 => {
                let expected = elements
                    .checked_mul(std::mem::size_of::<f32>())
                    .context("linear byte length overflow (f32)")?;
                ensure!(
                    bytes.len() == expected,
                    "snapshot tensor bytes {} do not match expected {} for F32",
                    bytes.len(),
                    expected
                );
                let mut values = vec![0f32; elements];
                decode_linear_bytes(bytes, &mut values, 4, parallel, |chunk| {
                    f32::from_le_bytes(chunk.try_into().expect("chunk len 4"))
                })?;
                Tensor::from_vec(values, (out_dim, in_dim), &Device::Cpu)?
            }
            DsqTensorDType::F16 => {
                let expected = elements
                    .checked_mul(2)
                    .context("linear byte length overflow (f16)")?;
                ensure!(
                    bytes.len() == expected,
                    "snapshot tensor bytes {} do not match expected {} for F16",
                    bytes.len(),
                    expected
                );
                let mut values = vec![f16::ZERO; elements];
                decode_linear_bytes(bytes, &mut values, 2, parallel, |chunk| {
                    let bits = u16::from_le_bytes(chunk.try_into().expect("chunk len 2"));
                    f16::from_bits(bits)
                })?;
                Tensor::from_vec(values, (out_dim, in_dim), &Device::Cpu)?
            }
            DsqTensorDType::BF16 => {
                let expected = elements
                    .checked_mul(2)
                    .context("linear byte length overflow (bf16)")?;
                ensure!(
                    bytes.len() == expected,
                    "snapshot tensor bytes {} do not match expected {} for BF16",
                    bytes.len(),
                    expected
                );
                let mut values = vec![bf16::ZERO; elements];
                decode_linear_bytes(bytes, &mut values, 2, parallel, |chunk| {
                    let bits = u16::from_le_bytes(chunk.try_into().expect("chunk len 2"));
                    bf16::from_bits(bits)
                })?;
                Tensor::from_vec(values, (out_dim, in_dim), &Device::Cpu)?
            }
            other => bail!(
                "snapshot dtype {:?} is not supported as float tensor",
                other
            ),
        };
        Ok(tensor.to_device(device)?)
    }

    fn bias_tensor(
        &self,
        bytes: &[u8],
        dtype: DsqBiasDType,
        out_dim: usize,
        device: &Device,
    ) -> Result<Tensor> {
        match dtype {
            DsqBiasDType::F16 => {
                ensure!(
                    bytes.len() == out_dim * std::mem::size_of::<f16>(),
                    "snapshot bias bytes {} do not match expected {} for F16",
                    bytes.len(),
                    out_dim * std::mem::size_of::<f16>()
                );
                let mut values = Vec::with_capacity(out_dim);
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes(chunk.try_into().expect("chunk len 2"));
                    values.push(f16::from_bits(bits));
                }
                Ok(Tensor::from_vec(values, out_dim, &Device::Cpu)?.to_device(device)?)
            }
            DsqBiasDType::F32 => {
                ensure!(
                    bytes.len() == out_dim * std::mem::size_of::<f32>(),
                    "snapshot bias bytes {} do not match expected {} for F32",
                    bytes.len(),
                    out_dim * std::mem::size_of::<f32>()
                );
                let mut values = Vec::with_capacity(out_dim);
                for chunk in bytes.chunks_exact(4) {
                    values.push(f32::from_le_bytes(chunk.try_into().expect("chunk len 4")));
                }
                Ok(Tensor::from_vec(values, out_dim, &Device::Cpu)?.to_device(device)?)
            }
            DsqBiasDType::BF16 => {
                ensure!(
                    bytes.len() == out_dim * std::mem::size_of::<bf16>(),
                    "snapshot bias bytes {} do not match expected {} for BF16",
                    bytes.len(),
                    out_dim * std::mem::size_of::<bf16>()
                );
                let mut values = Vec::with_capacity(out_dim);
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes(chunk.try_into().expect("chunk len 2"));
                    values.push(bf16::from_bits(bits));
                }
                Ok(Tensor::from_vec(values, out_dim, &Device::Cpu)?.to_device(device)?)
            }
            other => bail!("snapshot bias dtype {:?} is not supported", other),
        }
    }
}

fn ggml_from_snapshot_dtype(dtype: DsqTensorDType) -> Result<GgmlDType> {
    match dtype {
        DsqTensorDType::Q8_0 => Ok(GgmlDType::Q8_0),
        DsqTensorDType::Q4K => Ok(GgmlDType::Q4K),
        DsqTensorDType::Q6K => Ok(GgmlDType::Q6K),
        other => bail!("snapshot dtype {:?} does not map to ggml", other),
    }
}

fn ensure_supported_snapshot_dtype(dtype: DsqTensorDType) -> Result<()> {
    match dtype {
        DsqTensorDType::Q8_0 | DsqTensorDType::Q4K | DsqTensorDType::Q6K => Ok(()),
        DsqTensorDType::F16 | DsqTensorDType::BF16 | DsqTensorDType::F32 => Ok(()),
    }
}

fn resolve_snapshot_storage(path: &Path) -> Result<SnapshotStorage> {
    let snapshot_path = if path.is_file() {
        path.to_path_buf()
    } else if path.is_dir() {
        find_dsq_file(path)?.unwrap_or_else(|| path.join("snapshot.dsq"))
    } else {
        bail!("snapshot path {} does not exist", path.display());
    };
    if !is_dsq_file(&snapshot_path)? {
        bail!(
            "{} is not a .dsq file; pass a valid snapshot container",
            snapshot_path.display()
        );
    }
    Ok(SnapshotStorage::Dsq {
        path: snapshot_path,
    })
}

fn find_dsq_file(dir: &Path) -> Result<Option<PathBuf>> {
    let mut matches = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("failed to list {}", dir.display()))? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        if has_extension(&path, "dsq") {
            matches.push(path);
        }
    }
    if matches.is_empty() {
        Ok(None)
    } else if matches.len() == 1 {
        Ok(matches.pop())
    } else {
        bail!(
            "found multiple .dsq files under {}; pass an explicit file path",
            dir.display()
        );
    }
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(expected))
        .unwrap_or(false)
}

fn is_dsq_file(path: &Path) -> Result<bool> {
    if has_extension(path, "dsq") {
        return Ok(true);
    }
    if !path.is_file() {
        return Ok(false);
    }
    let mut file = File::open(path).with_context(|| {
        format!(
            "failed to open {} while probing snapshot magic",
            path.display()
        )
    })?;
    let mut magic = [0u8; 8];
    let read = file.read(&mut magic)?;
    if read >= SNAPSHOT_DSQ_MAGIC.len() {
        return Ok(&magic[..SNAPSHOT_DSQ_MAGIC.len()] == SNAPSHOT_DSQ_MAGIC);
    }
    Ok(false)
}

fn env_bool(var: &str) -> Option<bool> {
    env::var(var)
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

fn should_parallelize(plan: &SnapshotLoadPlan, cfg: &ParallelConfig) -> bool {
    if !cfg.enabled || cfg.max_jobs <= 1 || plan.len() < 2 {
        return false;
    }
    plan.linears()
        .iter()
        .filter_map(|spec| spec.elements())
        .any(|elements| should_parallelize_tensor(elements, cfg))
}

fn should_parallelize_tensor(elements: usize, cfg: &ParallelConfig) -> bool {
    cfg.enabled && cfg.max_jobs > 1 && elements >= cfg.min_tensor_elements
}

fn decode_linear_bytes<T, F>(
    bytes: &[u8],
    values: &mut [T],
    chunk: usize,
    cfg: &ParallelConfig,
    decode: F,
) -> Result<()>
where
    T: Send,
    F: Sync + Fn(&[u8]) -> T,
{
    let parallel = should_parallelize_tensor(values.len(), cfg);
    if parallel && current_thread_index().is_none() {
        let num_threads = cfg.max_jobs.max(1);
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .with_context(|| {
                format!(
                    "failed to build dense decode pool with {} threads",
                    num_threads
                )
            })?;
        pool.install(|| decode_linear_bytes_inner(bytes, values, chunk, &decode, true));
    } else {
        decode_linear_bytes_inner(bytes, values, chunk, &decode, parallel);
    }
    Ok(())
}

fn decode_linear_bytes_inner<T, F>(
    bytes: &[u8],
    values: &mut [T],
    chunk: usize,
    decode: &F,
    parallel: bool,
) where
    T: Send,
    F: Sync + Fn(&[u8]) -> T,
{
    if parallel {
        bytes
            .par_chunks_exact(chunk)
            .zip(values.par_iter_mut())
            .for_each(|(chunk_bytes, slot)| {
                *slot = decode(chunk_bytes);
            });
    } else {
        for (slot, chunk_bytes) in values.iter_mut().zip(bytes.chunks_exact(chunk)) {
            *slot = decode(chunk_bytes);
        }
    }
}
