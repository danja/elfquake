#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUTPUT="${OUTPUT:-data/derived/models/common_transformer_fixture.csv}"
REPORT="${REPORT:-data/derived/models/common_transformer_fixture.json}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-common-window-fixture \
  --input data/derived/models/mountain_256x256_seed40_20000.aligned_hourly_synthetic_windows.csv \
  --input data/derived/multimodal/all_italy.spatial_vlf_image_windows.labeled.csv \
  --input data/derived/models/japan_vlf_model_input.m5.csv \
  --dataset-id seed40 \
  --dataset-id italy_all \
  --dataset-id japan_moshiri \
  --out "$OUTPUT" --report "$REPORT"
