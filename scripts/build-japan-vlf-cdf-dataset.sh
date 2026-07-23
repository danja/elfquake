#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
FEATURE_ROOT="${FEATURE_ROOT:-data/derived/vlf/japan}"
WINDOWS="${WINDOWS:?Set WINDOWS to Japan seismic training windows CSV}"
OUTPUT="${OUTPUT:-data/derived/japan/japan.vlf_cdf_window_features.csv}"

features=()
while IFS= read -r path; do
  features+=("$path")
done < <(find "$FEATURE_ROOT" -maxdepth 1 -type f -name '*.features.csv' | sort)
[[ "${#features[@]}" -gt 0 ]] || { echo "No Japan CDF feature CSVs found" >&2; exit 2; }

feature_args=()
for feature in "${features[@]}"; do
  feature_args+=(--features "$feature")
done
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-japan-vlf-cdf-window-features \
  "${feature_args[@]}" --windows "$WINDOWS" --out "$OUTPUT"
echo "combined feature files: ${#features[@]}"
