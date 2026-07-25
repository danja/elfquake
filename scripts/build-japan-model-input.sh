#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WINDOWS="${WINDOWS:-data/derived/japan/japan.seismic_training_windows.csv}"
VLF_WINDOWS="${VLF_WINDOWS:-data/derived/japan/japan.vlf_cdf_window_features.csv}"
OUTPUT="${OUTPUT:-data/derived/models/japan_vlf_model_input.csv}"
DATASET_ID="${DATASET_ID:-japan_moshiri}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-japan-model-input \
  --windows "$WINDOWS" --vlf-windows "$VLF_WINDOWS" --out "$OUTPUT" --dataset-id "$DATASET_ID"
