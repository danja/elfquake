#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
FEATURES="${FEATURES:?Set FEATURES to a Japan CDF feature CSV}"
WINDOWS="${WINDOWS:?Set WINDOWS to UTC training windows CSV}"
OUTPUT="${OUTPUT:-data/derived/vlf/japan/window_features.csv}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-japan-vlf-cdf-window-features \
  --features "$FEATURES" --windows "$WINDOWS" --out "$OUTPUT"
