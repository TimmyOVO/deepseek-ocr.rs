use std::{
    convert::TryFrom,
    pin::Pin,
    sync::{Arc, Mutex},
};

use deepseek_ocr_core::streaming::DeltaTracker;
use rocket::{
    response::stream::{Event, EventStream},
    tokio::sync::mpsc,
};
use serde_json::json;
use tokenizers::Tokenizer;
use tokio_stream::wrappers::UnboundedReceiverStream;

pub type BoxEventStream =
    EventStream<Pin<Box<dyn rocket::futures::stream::Stream<Item = Event> + Send>>>;

pub fn into_event_stream(rx: mpsc::UnboundedReceiver<Event>) -> BoxEventStream {
    let stream = UnboundedReceiverStream::new(rx);
    let boxed: Pin<Box<dyn rocket::futures::stream::Stream<Item = Event> + Send>> =
        Box::pin(stream);
    EventStream::from(boxed)
}

#[derive(Clone)]
pub struct StreamContext {
    pub sender: mpsc::UnboundedSender<Event>,
    pub kind: StreamKind,
}

impl StreamContext {
    pub fn send_error(&self, message: &str) {
        match &self.kind {
            StreamKind::Responses { .. } => {
                let _ = self.sender.send(Event::json(&json!({
                    "type": "response.error",
                    "error": { "message": message },
                })));
                let _ = self.sender.send(Event::data("[DONE]"));
            }
            StreamKind::Chat {
                completion_id,
                model,
                created,
            } => {
                let payload = json!({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": serde_json::Value::Object(serde_json::Map::new()),
                        "finish_reason": "error",
                    }],
                    "error": { "message": message },
                });
                let _ = self.sender.send(Event::json(&payload));
                let _ = self.sender.send(Event::data("[DONE]"));
            }
        }
    }
}

#[derive(Clone)]
pub enum StreamKind {
    Responses {
        response_id: String,
        output_id: String,
        model: String,
        created: i64,
    },
    Chat {
        completion_id: String,
        model: String,
        created: i64,
    },
}

struct StreamControllerInner {
    sender: mpsc::UnboundedSender<Event>,
    tokenizer: Arc<Tokenizer>,
    kind: StreamKind,
    runtime: Mutex<StreamRuntime>,
}

#[derive(Default)]
struct StreamRuntime {
    last_count: usize,
    role_sent: bool,
    finished: bool,
    delta: DeltaTracker,
}

pub struct StreamController {
    inner: Arc<StreamControllerInner>,
}

impl StreamController {
    pub fn new(tokenizer: Arc<Tokenizer>, context: StreamContext) -> Self {
        StreamController {
            inner: Arc::new(StreamControllerInner {
                sender: context.sender,
                tokenizer,
                kind: context.kind,
                runtime: Mutex::new(StreamRuntime::default()),
            }),
        }
    }

    pub fn send_initial(&self) {
        self.inner.send_initial();
    }

    pub fn flush_remaining(&self, tokens: &[i64]) {
        self.inner.flush_remaining(tokens);
    }

    pub fn finalize(&self, normalized: &str, prompt_tokens: usize, completion_tokens: usize) {
        self.inner
            .finalize(normalized, prompt_tokens, completion_tokens);
    }

    pub fn callback(&self) -> impl Fn(usize, &[i64]) + Send + Sync + 'static {
        let inner = Arc::clone(&self.inner);
        move |count: usize, ids: &[i64]| {
            inner.handle_progress(count, ids);
        }
    }

    pub fn emit_fallback(&self, text: &str) {
        self.inner.emit_delta(text.to_string(), true);
        self.inner.finalize(text, 0, 0);
    }
}

impl StreamControllerInner {
    fn send_initial(&self) {
        match &self.kind {
            StreamKind::Responses {
                response_id,
                model,
                created,
                ..
            } => {
                let _ = self.sender.send(Event::json(&json!({
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created": created,
                        "model": model,
                    }
                })));
            }
            StreamKind::Chat {
                completion_id,
                model,
                created,
            } => {
                let payload = json!({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                        },
                        "finish_reason": serde_json::Value::Null,
                    }],
                });
                let _ = self.sender.send(Event::json(&payload));
                if let Ok(mut state) = self.runtime.lock() {
                    state.role_sent = true;
                }
            }
        }
    }

    fn decode_tokens(&self, ids: &[i64]) -> Option<String> {
        let tokens: Vec<u32> = ids
            .iter()
            .filter_map(|&id| u32::try_from(id).ok())
            .collect();
        if tokens.is_empty() {
            return None;
        }
        self.tokenizer
            .decode(&tokens, true)
            .ok()
            .filter(|s| !s.is_empty())
    }

    fn emit_delta(&self, text: String, include_role: bool) {
        match &self.kind {
            StreamKind::Responses {
                response_id,
                output_id,
                model,
                created,
            } => {
                let payload = json!({
                    "type": "response.output_text.delta",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created": created,
                        "model": model,
                    },
                    "output_id": output_id,
                    "output_index": 0,
                    "delta": text,
                });
                let _ = self.sender.send(Event::json(&payload));
            }
            StreamKind::Chat {
                completion_id,
                model,
                created,
            } => {
                let mut delta = json!({ "content": text });
                if include_role {
                    if let serde_json::Value::Object(obj) = &mut delta {
                        obj.insert("role".into(), serde_json::Value::String("assistant".into()));
                    }
                }
                let payload = json!({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": serde_json::Value::Null,
                    }],
                });
                let _ = self.sender.send(Event::json(&payload));
            }
        }
    }

    fn handle_progress(&self, count: usize, ids: &[i64]) {
        self.process_tokens(count, ids, false);
    }

    fn flush_remaining(&self, ids: &[i64]) {
        let len = ids.len();
        if len == 0 {
            return;
        }
        self.process_tokens(len, ids, true);
    }

    fn process_tokens(&self, count: usize, ids: &[i64], is_final: bool) {
        if count == 0 {
            return;
        }

        let mut maybe_delta = None;
        let mut include_role_flag = false;

        {
            let mut state = self.runtime.lock().expect("stream state lock poisoned");

            if count <= state.last_count {
                state.last_count = count;
                return;
            }

            let include_role = matches!(self.kind, StreamKind::Chat { .. }) && !state.role_sent;

            if let Some(full_text) = self.decode_tokens(&ids[..count]) {
                let delta = state.delta.advance(&full_text, is_final);
                let should_emit = include_role || !delta.is_empty();

                if should_emit {
                    include_role_flag = include_role;
                    maybe_delta = Some(delta);
                    if include_role {
                        state.role_sent = true;
                    }
                }
            }

            state.last_count = count;
        }

        if let Some(delta) = maybe_delta {
            self.emit_delta(delta, include_role_flag);
        }
    }

    fn finalize(&self, normalized: &str, prompt_tokens: usize, completion_tokens: usize) {
        {
            let mut state = self.runtime.lock().expect("stream state lock poisoned");
            if state.finished {
                return;
            }
            state.finished = true;
        }

        match &self.kind {
            StreamKind::Responses {
                response_id,
                output_id,
                model,
                created,
            } => {
                let total_tokens = prompt_tokens + completion_tokens;
                let payload = json!({
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created": created,
                        "model": model,
                        "output": [{
                            "id": output_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": normalized,
                            }],
                        }],
                        "usage": {
                            "input_tokens": prompt_tokens,
                            "output_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        },
                    }
                });
                let _ = self.sender.send(Event::json(&payload));
                let _ = self.sender.send(Event::data("[DONE]"));
            }
            StreamKind::Chat {
                completion_id,
                model,
                created,
            } => {
                let payload = json!({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": serde_json::Value::Object(serde_json::Map::new()),
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                });
                let _ = self.sender.send(Event::json(&payload));
                let _ = self.sender.send(Event::data("[DONE]"));
            }
        }
    }
}
