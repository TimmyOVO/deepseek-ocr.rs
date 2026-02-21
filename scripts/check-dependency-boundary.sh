#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET_DIRS=(
  "crates/cli/src"
  "crates/server/src"
)

FORBIDDEN_CRATES=(
  "deepseek_ocr_core"
  "deepseek_ocr_config"
  "deepseek_ocr_assets"
  "deepseek_ocr_infer_deepseek"
  "deepseek_ocr_infer_paddleocr"
  "deepseek_ocr_infer_dots"
  "deepseek_ocr_infer_glm"
)

echo "Checking dependency boundaries for CLI and server entrypoints..."

violations=()

for dir in "${TARGET_DIRS[@]}"; do
  abs_dir="${ROOT_DIR}/${dir}"
  [ -d "${abs_dir}" ] || continue

  for crate in "${FORBIDDEN_CRATES[@]}"; do
    while IFS= read -r line; do
      violations+=("${line}")
    done < <(grep -R -n -E "^[[:space:]]*(pub[[:space:]]+)?use[[:space:]]+${crate}::" "${abs_dir}" || true)
  done
done

if [ ${#violations[@]} -eq 0 ]; then
  echo "OK: no forbidden imports found in ${TARGET_DIRS[*]}"
  exit 0
fi

echo "Forbidden imports detected (these must route through deepseek_ocr_pipeline instead):"
printf '%s\n' "${violations[@]}"
exit 1
