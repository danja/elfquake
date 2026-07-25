#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
FEATURE_ROOT="${FEATURE_ROOT:-data/derived/vlf/japan}"
OUT_ROOT="${OUT_ROOT:-data/derived/models/japan_moshiri_sequences}"
mkdir -p "$OUT_ROOT"

features=()
while IFS= read -r path; do features+=("$path"); done < <(
  find "$FEATURE_ROOT" -maxdepth 1 -type f -name '*.features.csv' | sort
)
[[ "${#features[@]}" -gt 0 ]] || { echo "No Japan CDF feature CSVs found" >&2; exit 2; }

manifests=()
for feature in "${features[@]}"; do
  stem="$(basename "$feature" .features.csv)"
  out_dir="$OUT_ROOT/${stem}_sequence"
  dataset_id="japan_moshiri_${stem}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli \
    materialize-japan-vlf-cdf-sequence --features "$feature" \
    --out-dir "$out_dir" --dataset-id "$dataset_id"
  manifests+=("$out_dir/manifest.json")
done

printf '%s\n' "${manifests[@]}" > "$OUT_ROOT/manifests.txt"
echo "manifest index: $OUT_ROOT/manifests.txt"
