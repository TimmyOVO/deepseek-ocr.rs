use anyhow::{bail, Result};
use deepseek_ocr_dsq::DsqTensorDType;
use serde_json::Value;

mod adapters;

pub use adapters::{DeepSeekOcrAdapter, DotsOcrAdapter, PaddleOcrVlAdapter};

#[derive(Debug, Clone)]
pub struct LinearSpec {
    pub name: String,
    pub out_dim: usize,
    pub in_dim: usize,
    pub bias: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum AdapterScope {
    Text,
    TextAndProjector,
}

impl AdapterScope {
    pub fn includes_projector(self) -> bool {
        matches!(self, Self::TextAndProjector)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuantContext {
    pub primary: DsqTensorDType,
}

pub trait ModelAdapter: Sync {
    fn id(&self) -> &'static str;
    fn supports(&self, cfg: &Value) -> bool;
    fn discover(&self, cfg: &Value, scope: AdapterScope) -> Result<Vec<LinearSpec>>;
    fn recommend_dtype(
        &self,
        _tensor: &str,
        _in_dim: usize,
        _ctx: &QuantContext,
    ) -> Option<DsqTensorDType> {
        None
    }
}

pub struct AdapterRegistry {
    adapters: &'static [&'static dyn ModelAdapter],
}

impl AdapterRegistry {
    pub const fn new(adapters: &'static [&'static dyn ModelAdapter]) -> Self {
        Self { adapters }
    }

    pub fn global() -> &'static Self {
        &REGISTRY
    }

    pub fn list(&self) -> &'static [&'static dyn ModelAdapter] {
        self.adapters
    }

    pub fn infer_adapter(&self, cfg: &Value) -> Result<&'static dyn ModelAdapter> {
        let matches: Vec<&'static dyn ModelAdapter> = self
            .adapters
            .iter()
            .copied()
            .filter(|adapter| adapter.supports(cfg))
            .collect();
        match matches.len() {
            1 => Ok(matches[0]),
            0 => bail!("no registered adapters support the provided config"),
            _ => {
                let ids = matches
                    .iter()
                    .map(|adapter| adapter.id())
                    .collect::<Vec<_>>()
                    .join(", ");
                bail!("multiple adapters match the provided config ({ids}); please pass --adapter to disambiguate");
            }
        }
    }

    pub fn get(&self, id: &str) -> Option<&'static dyn ModelAdapter> {
        self.adapters
            .iter()
            .copied()
            .find(|adapter| adapter.id() == id)
    }
}

use adapters::{
    DeepSeekOcrAdapter as DeepSeekAdapterType, DotsOcrAdapter as DotsAdapterType,
    PaddleOcrVlAdapter as PaddleAdapterType,
};

static DEEPSEEK_OCR_ADAPTER: DeepSeekAdapterType = DeepSeekAdapterType;
static PADDLE_OCR_VL_ADAPTER: PaddleAdapterType = PaddleAdapterType;
static DOTS_OCR_ADAPTER: DotsAdapterType = DotsAdapterType;
static REGISTERED_ADAPTERS: [&'static dyn ModelAdapter; 3] = [
    &DEEPSEEK_OCR_ADAPTER,
    &PADDLE_OCR_VL_ADAPTER,
    &DOTS_OCR_ADAPTER,
];
static REGISTRY: AdapterRegistry = AdapterRegistry::new(&REGISTERED_ADAPTERS);
