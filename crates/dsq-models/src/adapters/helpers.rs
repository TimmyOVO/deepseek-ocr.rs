use anyhow::{Context, Result};
use serde_json::{Map, Value};

pub(crate) fn root_object(value: &Value) -> Result<&Map<String, Value>> {
    value
        .as_object()
        .context("config JSON must contain a top-level object")
}

pub(crate) fn get_required_usize(map: &Map<String, Value>, key: &str) -> Result<usize> {
    get_optional_usize(map, key).with_context(|| format!("missing numeric field `{key}`"))
}

pub(crate) fn get_optional_usize(map: &Map<String, Value>, key: &str) -> Option<usize> {
    map.get(key).and_then(value_to_usize)
}

pub(crate) fn get_optional_nonzero(map: &Map<String, Value>, key: &str) -> Option<usize> {
    get_optional_usize(map, key).and_then(|value| if value == 0 { None } else { Some(value) })
}

pub(crate) fn value_to_usize(value: &Value) -> Option<usize> {
    match value {
        Value::Number(num) => num.as_u64().map(|v| v as usize).or_else(|| {
            num.as_i64()
                .and_then(|v| if v >= 0 { Some(v as usize) } else { None })
        }),
        Value::String(s) => s.parse::<usize>().ok(),
        _ => None,
    }
}
