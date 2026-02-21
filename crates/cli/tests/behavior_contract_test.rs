use std::{
    env,
    path::{Path, PathBuf},
    process::{Command, Output},
};

#[cfg(feature = "cli-debug")]
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

fn cli_bin() -> PathBuf {
    if let Some(path) = env::var_os("CARGO_BIN_EXE_deepseek-ocr-cli") {
        return PathBuf::from(path);
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fallback = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("workspace root must exist")
        .join("target")
        .join("debug")
        .join("deepseek-ocr-cli");
    fallback
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root must exist")
        .to_path_buf()
}

fn fixture_image() -> PathBuf {
    workspace_root()
        .join("baselines")
        .join("sample")
        .join("images")
        .join("test.png")
}

fn has_local_deepseek_weights() -> bool {
    workspace_root()
        .join("DeepSeek-OCR")
        .join("model-00001-of-000001.safetensors")
        .exists()
}

fn run_cli(args: &[&str]) -> Output {
    Command::new(cli_bin())
        .args(args)
        .output()
        .expect("failed to execute deepseek-ocr-cli")
}

fn assert_exit_code_1(output: &Output) {
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected exit code 1, got status={:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn missing_prompt_exits_with_required_prompt_error() {
    let image = fixture_image();
    let output = run_cli(&["--image", image.to_str().expect("utf-8 path")]);
    assert_exit_code_1(&output);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("prompt is required"),
        "stderr must contain missing prompt contract\nstderr:\n{stderr}"
    );
}

#[test]
fn mismatched_image_slots_exits_with_contract_message() {
    if !has_local_deepseek_weights() {
        return;
    }

    let output = run_cli(&[
        "--quiet",
        "--max-new-tokens",
        "0",
        "--prompt",
        "<image>\n<|grounding|>Convert the document to markdown.",
    ]);
    assert_exit_code_1(&output);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("expects") && stderr.contains("<image>"),
        "stderr must mention slot mismatch contract\nstderr:\n{stderr}"
    );
}

#[cfg(not(feature = "bench-metrics"))]
#[test]
fn bench_flag_without_feature_exits_with_actionable_error() {
    let output = run_cli(&["--bench"]);
    assert_exit_code_1(&output);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--features bench-metrics"),
        "stderr must mention feature gate contract\nstderr:\n{stderr}"
    );
}

#[cfg(feature = "bench-metrics")]
#[test]
fn bench_flag_without_feature_exits_with_actionable_error() {}

#[cfg(feature = "cli-debug")]
#[test]
fn debug_output_json_writes_stable_schema_keys() {
    if !has_local_deepseek_weights() {
        return;
    }

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock must be monotonic")
        .as_nanos();
    let output_path = env::temp_dir().join(format!("deepseek-ocr-cli-contract-{nanos}.json"));
    let image = fixture_image();

    let output = run_cli(&[
        "--quiet",
        "--max-new-tokens",
        "0",
        "--prompt",
        "<image>\n<|grounding|>Convert the document to markdown.",
        "--image",
        image.to_str().expect("utf-8 path"),
        "--output-json",
        output_path.to_str().expect("utf-8 path"),
    ]);

    assert!(
        output.status.success(),
        "expected successful run for debug json contract\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.exists(), "expected output json to be created");

    let bytes = fs::read(&output_path).expect("must read output json");
    let value: serde_json::Value = serde_json::from_slice(&bytes).expect("must parse output json");

    assert_eq!(value["schema_version"], 1);
    for key in [
        "rendered_prompt",
        "prompt_tokens",
        "generated_len",
        "tokens",
        "decoded",
        "normalized",
    ] {
        assert!(value.get(key).is_some(), "missing required key `{key}`");
    }

    let _ = fs::remove_file(output_path);
}
