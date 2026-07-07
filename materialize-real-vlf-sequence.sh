#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/multimodal/cumiana_last_E_VLF.image_features.csv}"
OUT_DIR="${OUT_DIR:-data/derived/models/cumiana_vlf_image_sequence}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli materialize-sequence-dataset \
  --input "$INPUT" \
  --out-dir "$OUT_DIR" \
  --time-field vlf_image_captured_at_utc \
  --no-entity-field \
  --modality real_vlf_image
