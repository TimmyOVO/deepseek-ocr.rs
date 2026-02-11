use std::{
    collections::HashMap,
    fmt,
    fs::File,
    io::{Cursor, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use thiserror::Error;

const DSQ_MAGIC: &[u8; 7] = b"DSQSNAP";
const DSQ_VERSION: u32 = 1;

/// Errors surfaced while reading or validating a `.dsq` container.
#[derive(Debug, Error)]
pub enum DsqError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid snapshot magic: found {found:?}")]
    InvalidMagic { found: [u8; 7] },
    #[error("unsupported snapshot version {found}, expected {expected}")]
    UnsupportedVersion { found: u32, expected: u32 },
    #[error("snapshot malformed: {0}")]
    Format(String),
    #[error("snapshot validation failed: {0}")]
    Validation(String),
    #[error("invalid UTF-8 string: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
}

/// Result alias for DSQ operations.
pub type Result<T> = std::result::Result<T, DsqError>;

/// Header metadata describing a DSQ snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsqHeader {
    pub version: u32,
    pub candle_version: String,
    pub model_id: String,
    pub backend: String,
    pub default_qdtype: DsqTensorDType,
    pub block_size: u32,
    pub tensor_count: u32,
}

/// Quantized tensor dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsqTensorDType {
    Q8_0,
    Q4K,
    Q6K,
    F16,
    BF16,
    F32,
}

impl DsqTensorDType {
    pub fn as_u32(self) -> u32 {
        match self {
            Self::Q8_0 => 8,
            Self::Q4K => 12,
            Self::Q6K => 14,
            Self::F16 => 1,
            Self::BF16 => 16,
            Self::F32 => 0,
        }
    }

    pub fn block_size(self) -> Option<usize> {
        match self {
            Self::Q8_0 => Some(32),
            Self::Q4K => Some(256),
            Self::Q6K => Some(256),
            Self::F16 | Self::BF16 | Self::F32 => None,
        }
    }

    pub fn elem_size_bytes(self) -> Option<usize> {
        match self {
            Self::F16 | Self::BF16 => Some(2),
            Self::F32 => Some(4),
            _ => None,
        }
    }

    pub fn is_quantized(self) -> bool {
        self.block_size().is_some()
    }
}

impl TryFrom<u32> for DsqTensorDType {
    type Error = DsqError;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            8 => Ok(Self::Q8_0),
            12 => Ok(Self::Q4K),
            14 => Ok(Self::Q6K),
            1 => Ok(Self::F16),
            16 => Ok(Self::BF16),
            0 => Ok(Self::F32),
            other => Err(DsqError::Format(format!(
                "unsupported tensor dtype code {other}"
            ))),
        }
    }
}

impl fmt::Display for DsqTensorDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Q8_0 => f.write_str("Q8_0"),
            Self::Q4K => f.write_str("Q4_K"),
            Self::Q6K => f.write_str("Q6_K"),
            Self::F16 => f.write_str("F16"),
            Self::BF16 => f.write_str("BF16"),
            Self::F32 => f.write_str("F32"),
        }
    }
}

/// Bias tensor dtype stored inside the container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsqBiasDType {
    U8,
    U32,
    I64,
    F16,
    F32,
    F64,
    BF16,
}

impl DsqBiasDType {
    pub fn as_u32(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::U32 => 1,
            Self::I64 => 2,
            Self::F16 => 3,
            Self::F32 => 4,
            Self::F64 => 5,
            Self::BF16 => 6,
        }
    }
}

impl TryFrom<u32> for DsqBiasDType {
    type Error = DsqError;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::U8),
            1 => Ok(Self::U32),
            2 => Ok(Self::I64),
            3 => Ok(Self::F16),
            4 => Ok(Self::F32),
            5 => Ok(Self::F64),
            6 => Ok(Self::BF16),
            other => Err(DsqError::Format(format!(
                "unsupported bias dtype code {other}"
            ))),
        }
    }
}

impl fmt::Display for DsqBiasDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U8 => f.write_str("U8"),
            Self::U32 => f.write_str("U32"),
            Self::I64 => f.write_str("I64"),
            Self::F16 => f.write_str("F16"),
            Self::F32 => f.write_str("F32"),
            Self::F64 => f.write_str("F64"),
            Self::BF16 => f.write_str("BF16"),
        }
    }
}

/// Record describing a single quantized tensor entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsqRecord {
    pub name: String,
    pub out_dim: usize,
    pub in_dim: usize,
    pub q_dtype: DsqTensorDType,
    pub q_offset: u64,
    pub q_len: u64,
    pub bias_offset: Option<u64>,
    pub bias_len: Option<u64>,
    pub bias_dtype: Option<DsqBiasDType>,
}

/// Reader that mmaps the DSQ payload and exposes zero-copy slices.
#[derive(Debug)]
pub struct DsqReader {
    path: PathBuf,
    data: Arc<Mmap>,
    header: DsqHeader,
    records: Vec<DsqRecord>,
    index: HashMap<String, usize>,
}

impl DsqReader {
    /// Open a DSQ file from disk, validating header and record constraints.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(path, Arc::new(mmap))
    }

    fn from_mmap(path: PathBuf, data: Arc<Mmap>) -> Result<Self> {
        let ParsedIndex {
            header,
            records,
            metadata_len,
        } = parse_index(&data)?;
        validate_header(&header)?;
        validate_records(&header, &records, metadata_len, data.len())?;
        let mut index = HashMap::new();
        for (idx, record) in records.iter().enumerate() {
            if index.insert(record.name.clone(), idx).is_some() {
                return Err(DsqError::Validation(format!(
                    "duplicate tensor record `{}`",
                    record.name
                )));
            }
        }
        Ok(Self {
            path,
            data,
            header,
            records,
            index,
        })
    }

    /// Absolute path backing this reader.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Header metadata for the snapshot.
    pub fn header(&self) -> &DsqHeader {
        &self.header
    }

    /// All tensor records in the order encoded by the file.
    pub fn records(&self) -> &[DsqRecord] {
        &self.records
    }

    /// Lookup a tensor record by its fully qualified name.
    pub fn tensor(&self, name: &str) -> Option<&DsqRecord> {
        self.index.get(name).map(|&idx| &self.records[idx])
    }

    /// Obtain the quantized payload bytes for a record.
    pub fn tensor_bytes(&self, record: &DsqRecord) -> Result<&[u8]> {
        self.slice(record.q_offset, record.q_len)
    }

    /// Convenience helper returning the payload slice for the given tensor name.
    pub fn tensor_bytes_by_name(&self, name: &str) -> Result<Option<&[u8]>> {
        match self.tensor(name) {
            Some(record) => self.tensor_bytes(record).map(Some),
            None => Ok(None),
        }
    }

    /// Obtain the optional bias payload for a record.
    pub fn bias_bytes(&self, record: &DsqRecord) -> Result<Option<&[u8]>> {
        match (record.bias_offset, record.bias_len) {
            (Some(offset), Some(len)) => self.slice(offset, len).map(Some),
            (None, None) => Ok(None),
            _ => Err(DsqError::Validation(format!(
                "bias metadata for `{}` is inconsistent",
                record.name
            ))),
        }
    }

    fn slice(&self, offset: u64, len: u64) -> Result<&[u8]> {
        let start = usize::try_from(offset).map_err(|_| {
            DsqError::Validation(format!("offset {offset} does not fit in platform usize"))
        })?;
        let slice_len = usize::try_from(len).map_err(|_| {
            DsqError::Validation(format!("length {len} does not fit in platform usize"))
        })?;
        let end = start
            .checked_add(slice_len)
            .ok_or_else(|| DsqError::Validation(format!("slice {offset}+{len} overflows usize")))?;
        if end > self.data.len() {
            return Err(DsqError::Validation(format!(
                "slice [{offset}, {offset}+{len}) exceeds file size {}",
                self.data.len()
            )));
        }
        Ok(&self.data[start..end])
    }
}

struct ParsedIndex {
    header: DsqHeader,
    records: Vec<DsqRecord>,
    metadata_len: usize,
}

fn parse_index(bytes: &[u8]) -> Result<ParsedIndex> {
    let mut cursor = Cursor::new(bytes);
    let mut magic = [0u8; DSQ_MAGIC.len()];
    cursor.read_exact(&mut magic)?;
    if &magic != DSQ_MAGIC {
        return Err(DsqError::InvalidMagic { found: magic });
    }
    let version = cursor.read_u32::<LittleEndian>()?;
    if version != DSQ_VERSION {
        return Err(DsqError::UnsupportedVersion {
            found: version,
            expected: DSQ_VERSION,
        });
    }
    let candle_version = read_string(&mut cursor)?;
    let model_id = read_string(&mut cursor)?;
    let backend = read_string(&mut cursor)?;
    let default_qdtype_raw = cursor.read_u32::<LittleEndian>()?;
    let default_qdtype = DsqTensorDType::try_from(default_qdtype_raw)?;
    let block_size = cursor.read_u32::<LittleEndian>()?;
    if block_size == 0 {
        return Err(DsqError::Validation(
            "block_size must be non-zero".to_string(),
        ));
    }
    let tensor_count = cursor.read_u32::<LittleEndian>()?;
    let mut records = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut cursor)?;
        let out_dim = cursor.read_u32::<LittleEndian>()? as usize;
        let in_dim = cursor.read_u32::<LittleEndian>()? as usize;
        let q_dtype_raw = cursor.read_u32::<LittleEndian>()?;
        let q_dtype = DsqTensorDType::try_from(q_dtype_raw)?;
        let q_offset = cursor.read_u64::<LittleEndian>()?;
        let q_len = cursor.read_u64::<LittleEndian>()?;
        let bias_offset_raw = cursor.read_u64::<LittleEndian>()?;
        let bias_len_raw = cursor.read_u64::<LittleEndian>()?;
        let bias_dtype_raw = cursor.read_u32::<LittleEndian>()?;
        let (bias_offset, bias_len, bias_dtype) = if bias_len_raw == 0 {
            (None, None, None)
        } else {
            let dtype = DsqBiasDType::try_from(bias_dtype_raw)?;
            (Some(bias_offset_raw), Some(bias_len_raw), Some(dtype))
        };
        records.push(DsqRecord {
            name,
            out_dim,
            in_dim,
            q_dtype,
            q_offset,
            q_len,
            bias_offset,
            bias_len,
            bias_dtype,
        });
    }
    let metadata_len = usize::try_from(cursor.position())
        .map_err(|_| DsqError::Format("metadata exceeds addressable range".into()))?;
    if records.len() != tensor_count as usize {
        return Err(DsqError::Format(format!(
            "expected {tensor_count} records, parsed {}",
            records.len()
        )));
    }
    Ok(ParsedIndex {
        header: DsqHeader {
            version,
            candle_version,
            model_id,
            backend,
            default_qdtype,
            block_size,
            tensor_count,
        },
        records,
        metadata_len,
    })
}

fn validate_header(header: &DsqHeader) -> Result<()> {
    let expected = header.default_qdtype.block_size().ok_or_else(|| {
        DsqError::Validation(format!(
            "snapshot dtype {:?} not supported",
            header.default_qdtype
        ))
    })?;
    if header.block_size as usize != expected {
        return Err(DsqError::Validation(format!(
            "snapshot block size {} mismatches expected {} for {:?}",
            header.block_size, expected, header.default_qdtype
        )));
    }
    Ok(())
}

fn validate_records(
    _header: &DsqHeader,
    records: &[DsqRecord],
    metadata_len: usize,
    total_len: usize,
) -> Result<()> {
    if total_len < metadata_len {
        return Err(DsqError::Validation(
            "file smaller than metadata region".into(),
        ));
    }
    for record in records {
        if record.q_len == 0 {
            return Err(DsqError::Validation(format!(
                "tensor `{}` has empty quantized payload",
                record.name
            )));
        }
        if record.q_offset < metadata_len as u64 {
            return Err(DsqError::Validation(format!(
                "tensor `{}` q_offset {} overlaps metadata ({} bytes)",
                record.name, record.q_offset, metadata_len
            )));
        }
        check_bounds(
            record.q_offset,
            record.q_len,
            total_len,
            &record.name,
            "quantized",
        )?;
        if let Some(offset) = record.bias_offset {
            let len = record.bias_len.ok_or_else(|| {
                DsqError::Validation(format!(
                    "tensor `{}` bias offset present but length missing",
                    record.name
                ))
            })?;
            check_bounds(offset, len, total_len, &record.name, "bias")?;
        } else if record.bias_len.is_some() || record.bias_dtype.is_some() {
            return Err(DsqError::Validation(format!(
                "tensor `{}` bias metadata inconsistent",
                record.name
            )));
        }
        if let Some(rblock) = record.q_dtype.block_size() {
            if record.in_dim % rblock != 0 {
                return Err(DsqError::Validation(format!(
                    "tensor `{}` in_dim {} not divisible by block_size {} (dtype {:?})",
                    record.name, record.in_dim, rblock, record.q_dtype
                )));
            }
        } else if let Some(elem_size) = record.q_dtype.elem_size_bytes() {
            let elements = record.out_dim.checked_mul(record.in_dim).ok_or_else(|| {
                DsqError::Validation(format!(
                    "tensor `{}` dimensions overflow when validating float payload",
                    record.name
                ))
            })?;
            let expected_len = elements.checked_mul(elem_size).ok_or_else(|| {
                DsqError::Validation(format!(
                    "tensor `{}` byte length overflows for dtype {:?}",
                    record.name, record.q_dtype
                ))
            })?;
            let expected_len = u64::try_from(expected_len).map_err(|_| {
                DsqError::Validation(format!(
                    "tensor `{}` expected byte length exceeds u64",
                    record.name
                ))
            })?;
            if record.q_len != expected_len {
                return Err(DsqError::Validation(format!(
                    "tensor `{}` has q_len {} but expected {} bytes for {:?}",
                    record.name, record.q_len, expected_len, record.q_dtype
                )));
            }
        } else {
            return Err(DsqError::Validation(format!(
                "tensor `{}` has unsupported dtype {:?}",
                record.name, record.q_dtype
            )));
        }
    }
    Ok(())
}

fn check_bounds(offset: u64, len: u64, total_len: usize, tensor: &str, label: &str) -> Result<()> {
    let start = usize::try_from(offset).map_err(|_| {
        DsqError::Validation(format!(
            "tensor `{tensor}` {label} offset {offset} exceeds platform usize"
        ))
    })?;
    let slice_len = usize::try_from(len).map_err(|_| {
        DsqError::Validation(format!(
            "tensor `{tensor}` {label} length {len} exceeds platform usize"
        ))
    })?;
    let end = start.checked_add(slice_len).ok_or_else(|| {
        DsqError::Validation(format!(
            "tensor `{tensor}` {label} slice {offset}+{len} overflows usize"
        ))
    })?;
    if end > total_len {
        return Err(DsqError::Validation(format!(
            "tensor `{tensor}` {label} slice [{offset}, {offset}+{len}) exceeds file size {}",
            total_len
        )));
    }
    Ok(())
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
    let len = cursor.read_u32::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor
        .read_exact(&mut buf)
        .map_err(|err| DsqError::Format(format!("truncated string: {err}")))?;
    Ok(String::from_utf8(buf)?)
}
