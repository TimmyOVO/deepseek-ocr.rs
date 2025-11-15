use anyhow::{bail, Result};
use deepseek_ocr_dsq::DsqTensorDType;
use serde_json::Value;

mod adapters;

pub use adapters::{DeepSeekOcrAdapter, PaddleOcrVlAdapter};

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
    DeepSeekOcrAdapter as DeepSeekAdapterType, PaddleOcrVlAdapter as PaddleAdapterType,
};

static DEEPSEEK_OCR_ADAPTER: DeepSeekAdapterType = DeepSeekAdapterType;
static PADDLE_OCR_VL_ADAPTER: PaddleAdapterType = PaddleAdapterType;
static REGISTERED_ADAPTERS: [&'static dyn ModelAdapter; 2] =
    [&DEEPSEEK_OCR_ADAPTER, &PADDLE_OCR_VL_ADAPTER];
static REGISTRY: AdapterRegistry = AdapterRegistry::new(&REGISTERED_ADAPTERS);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deepseek_adapter_discovers_projector_and_lm_head() {
        let cfg = json!({
            "model_type": "deepseek_vl_v2",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "n_routed_experts": 0,
            "n_shared_experts": 0,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
            "lm_head": true,
            "vocab_size": 32,
            "projector_config": {
                "n_embed": 8,
                "input_dim": 4
            }
        });
        let adapter = AdapterRegistry::global()
            .infer_adapter(&cfg)
            .expect("infer adapter");
        assert_eq!(adapter.id(), "deepseek-ocr");
        let specs = adapter
            .discover(&cfg, AdapterScope::TextAndProjector)
            .expect("discover specs");
        assert!(specs.iter().any(|s| s.name == "lm_head.weight"));
        assert!(specs
            .iter()
            .any(|s| s.name == "model.projector.layers.weight"));
    }

    #[test]
    fn paddle_adapter_detects_from_model_type() {
        let cfg = json!({
            "model_type": "paddleocr_vl",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "vocab_size": 32000,
            "vision_config": {
                "hidden_size": 512,
                "intermediate_size": 1024,
                "num_hidden_layers": 2,
                "num_attention_heads": 8
            }
        });
        let adapter = AdapterRegistry::global()
            .infer_adapter(&cfg)
            .expect("infer adapter");
        assert_eq!(adapter.id(), "paddleocr-vl");
        let specs = adapter
            .discover(&cfg, AdapterScope::TextAndProjector)
            .expect("discover specs");
        assert!(specs.iter().any(|s| s
            .name
            .contains("visual.vision_model.encoder.layers.0.self_attn.q_proj")));
    }
}
