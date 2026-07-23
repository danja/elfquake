#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
FEATURES="${FEATURES:?Set FEATURES to one or more Japan CDF feature CSVs, separated by commas}"
WINDOWS="${WINDOWS:?Set WINDOWS to UTC training windows CSV}"
OUTPUT="${OUTPUT:-data/derived/vlf/japan/window_features.csv}"

feature_args=()
IFS=',' read -ra feature_paths <<< "$FEATURES"
for feature in "${feature_paths[@]}"; do
  feature_args+=(--features "$feature")
done
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-japan-vlf-cdf-window-features \
  "${feature_args[@]}" --windows "$WINDOWS" --out "$OUTPUT"
