use std::sync::Arc;

use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::cache::{DynamicCache, KvCacheChunk, KvCacheEntry};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeBackendKind {
    Cpu,
    Metal,
    Cuda,
}

impl RuntimeBackendKind {
    pub fn from_device(device: &Device) -> Self {
        if device.is_cuda() {
            Self::Cuda
        } else if device.is_metal() {
            Self::Metal
        } else {
            Self::Cpu
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCaps {
    pub backend: RuntimeBackendKind,
    pub paged_kv: bool,
    pub fused_attention: bool,
    pub fused_mlp: bool,
    pub prefers_static_workspace: bool,
}

impl BackendCaps {
    pub fn for_device(device: &Device) -> Self {
        Self::for_kind(RuntimeBackendKind::from_device(device))
    }

    pub fn for_kind(kind: RuntimeBackendKind) -> Self {
        match kind {
            RuntimeBackendKind::Cpu => Self {
                backend: kind,
                paged_kv: false,
                fused_attention: false,
                fused_mlp: false,
                prefers_static_workspace: false,
            },
            RuntimeBackendKind::Metal => Self {
                backend: kind,
                paged_kv: false,
                fused_attention: false,
                fused_mlp: false,
                prefers_static_workspace: true,
            },
            RuntimeBackendKind::Cuda => Self {
                backend: kind,
                paged_kv: true,
                fused_attention: true,
                fused_mlp: false,
                prefers_static_workspace: true,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeModelSpec {
    pub layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RopeState {
    pub prepared_len: usize,
}

impl RopeState {
    pub fn ensure_len(&mut self, len: usize) {
        self.prepared_len = self.prepared_len.max(len);
    }

    pub fn reset(&mut self) {
        self.prepared_len = 0;
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceArena {
    pub reserved_bytes: usize,
}

impl WorkspaceArena {
    pub fn reserve_bytes(&mut self, bytes: usize) {
        self.reserved_bytes = self.reserved_bytes.max(bytes);
    }

    pub fn clear(&mut self) {
        self.reserved_bytes = 0;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KvReadView<'a> {
    pub chunks: &'a [KvCacheChunk],
    pub total_len: usize,
}

#[derive(Debug, Clone)]
pub enum KvStateBackend {
    Generic { cache: DynamicCache },
    Metal { cache: DynamicCache },
    Cuda { cache: DynamicCache },
}

impl KvStateBackend {
    pub fn from_device(device: &Device, num_layers: usize) -> Self {
        let cache = DynamicCache::with_num_layers(num_layers);
        if device.is_cuda() {
            Self::Cuda { cache }
        } else if device.is_metal() {
            Self::Metal { cache }
        } else {
            Self::Generic { cache }
        }
    }

    pub fn backend_kind(&self) -> RuntimeBackendKind {
        match self {
            Self::Generic { .. } => RuntimeBackendKind::Cpu,
            Self::Metal { .. } => RuntimeBackendKind::Metal,
            Self::Cuda { .. } => RuntimeBackendKind::Cuda,
        }
    }

    fn cache(&self) -> &DynamicCache {
        match self {
            Self::Generic { cache } | Self::Metal { cache } | Self::Cuda { cache } => cache,
        }
    }

    fn cache_mut(&mut self) -> &mut DynamicCache {
        match self {
            Self::Generic { cache } | Self::Metal { cache } | Self::Cuda { cache } => cache,
        }
    }

    pub fn ensure_layers(&mut self, total_layers: usize) {
        self.cache_mut().ensure_layers(total_layers);
    }

    pub fn append(&mut self, layer_idx: usize, chunk: KvCacheChunk) -> Result<()> {
        self.cache_mut().append(layer_idx, chunk)
    }

    pub fn get(&self, layer_idx: usize) -> Option<&KvCacheEntry> {
        self.cache().get(layer_idx)
    }

    pub fn read(&self, layer_idx: usize) -> Option<KvReadView<'_>> {
        self.get(layer_idx).map(|entry| KvReadView {
            chunks: entry.chunks(),
            total_len: entry.seq_len(),
        })
    }

    pub fn seq_len(&self) -> Option<usize> {
        self.cache().seq_len()
    }

    pub fn num_layers(&self) -> usize {
        self.cache().num_layers()
    }

    pub fn clear(&mut self) {
        self.cache_mut().clear();
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeState {
    pub kv: KvStateBackend,
    pub rope: RopeState,
    pub workspace: WorkspaceArena,
    pub seq_len: usize,
}

impl RuntimeState {
    pub fn new(device: &Device, num_layers: usize) -> Self {
        let kv = KvStateBackend::from_device(device, num_layers);
        let seq_len = kv.seq_len().unwrap_or(0);
        Self {
            kv,
            rope: RopeState::default(),
            workspace: WorkspaceArena::default(),
            seq_len,
        }
    }

    pub fn backend_kind(&self) -> RuntimeBackendKind {
        self.kv.backend_kind()
    }

    pub fn ensure_layers(&mut self, total_layers: usize) {
        self.kv.ensure_layers(total_layers);
    }

    pub fn layer_entry(&self, layer_idx: usize) -> Option<&KvCacheEntry> {
        self.kv.get(layer_idx)
    }

    pub fn layer_view(&self, layer_idx: usize) -> Option<KvReadView<'_>> {
        self.kv.read(layer_idx)
    }

    pub fn append_chunk(&mut self, layer_idx: usize, chunk: KvCacheChunk) -> Result<()> {
        self.kv.append(layer_idx, chunk)?;
        self.seq_len = self.kv.seq_len().unwrap_or(0);
        Ok(())
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn num_layers(&self) -> usize {
        self.kv.num_layers()
    }

    pub fn prompt_guard(&mut self) -> RuntimeStateGuard<'_> {
        RuntimeStateGuard::new(self)
    }

    pub fn prompt_scope_guard(&mut self) -> PromptScopeGuard<'_> {
        PromptScopeGuard::new(self)
    }

    pub fn clear(&mut self) {
        self.kv.clear();
        self.rope.reset();
        self.workspace.clear();
        self.seq_len = 0;
    }
}

pub struct PromptScopeGuard<'a> {
    state: &'a mut RuntimeState,
}

impl<'a> PromptScopeGuard<'a> {
    pub fn new(state: &'a mut RuntimeState) -> Self {
        Self { state }
    }

    pub fn state(&mut self) -> &mut RuntimeState {
        self.state
    }
}

impl Drop for PromptScopeGuard<'_> {
    fn drop(&mut self) {
        self.state.clear();
    }
}

pub type RuntimeStateGuard<'a> = PromptScopeGuard<'a>;

#[derive(Debug, Clone, Copy)]
pub struct RuntimeStepRequest<'a> {
    pub hidden_states: &'a Tensor,
    pub attention_mask: Option<&'a Tensor>,
    pub position_ids: Option<&'a Tensor>,
    pub use_cache: bool,
}

#[derive(Debug)]
pub struct RuntimeStepOutput {
    pub hidden_states: Tensor,
    pub present: Option<KvCacheChunk>,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingRequest<'a> {
    pub logits: &'a Tensor,
}

#[derive(Debug)]
pub struct SamplingOutput {
    pub token_ids: Tensor,
}

pub trait FusedAttentionHook: Send + Sync {
    fn try_prefill(
        &self,
        request: &RuntimeStepRequest<'_>,
        state: &RuntimeState,
    ) -> Result<Option<RuntimeStepOutput>> {
        let _ = (request, state);
        Ok(None)
    }

    fn try_decode(
        &self,
        request: &RuntimeStepRequest<'_>,
        state: &RuntimeState,
    ) -> Result<Option<RuntimeStepOutput>> {
        let _ = (request, state);
        Ok(None)
    }
}

pub trait FusedMlpHook: Send + Sync {
    fn try_forward(&self, hidden_states: &Tensor) -> Result<Option<Tensor>> {
        let _ = hidden_states;
        Ok(None)
    }
}

#[derive(Default)]
pub struct BackendHooks {
    pub fused_attention: Option<Arc<dyn FusedAttentionHook>>,
    pub fused_mlp: Option<Arc<dyn FusedMlpHook>>,
}

pub trait RuntimeBackend: Send {
    fn backend_kind(&self) -> RuntimeBackendKind;

    fn caps(&self) -> BackendCaps;

    fn prefill_step(
        &mut self,
        request: &RuntimeStepRequest<'_>,
        state: &mut RuntimeState,
    ) -> Result<RuntimeStepOutput>;

    fn decode_step(
        &mut self,
        request: &RuntimeStepRequest<'_>,
        state: &mut RuntimeState,
    ) -> Result<RuntimeStepOutput>;

    fn sample_step(&mut self, request: &SamplingRequest<'_>) -> Result<SamplingOutput>;
}

pub struct RuntimeEngine {
    device: Device,
    hooks: BackendHooks,
    backend: Box<dyn RuntimeBackend>,
}

impl RuntimeEngine {
    pub fn new(device: Device, backend: Box<dyn RuntimeBackend>) -> Self {
        Self {
            device,
            hooks: BackendHooks::default(),
            backend,
        }
    }

    pub fn with_hooks(mut self, hooks: BackendHooks) -> Self {
        self.hooks = hooks;
        self
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn caps(&self) -> BackendCaps {
        self.backend.caps()
    }

    pub fn backend_kind(&self) -> RuntimeBackendKind {
        self.backend.backend_kind()
    }

    pub fn prefill_step(
        &mut self,
        request: &RuntimeStepRequest<'_>,
        state: &mut RuntimeState,
    ) -> Result<RuntimeStepOutput> {
        if let Some(hook) = self.hooks.fused_attention.as_ref()
            && let Some(output) = hook.try_prefill(request, state)?
        {
            return Ok(output);
        }
        self.backend.prefill_step(request, state)
    }

    pub fn decode_step(
        &mut self,
        request: &RuntimeStepRequest<'_>,
        state: &mut RuntimeState,
    ) -> Result<RuntimeStepOutput> {
        if let Some(hook) = self.hooks.fused_attention.as_ref()
            && let Some(output) = hook.try_decode(request, state)?
        {
            return Ok(output);
        }
        self.backend.decode_step(request, state)
    }

    pub fn sample_step(&mut self, request: &SamplingRequest<'_>) -> Result<SamplingOutput> {
        self.backend.sample_step(request)
    }
}
