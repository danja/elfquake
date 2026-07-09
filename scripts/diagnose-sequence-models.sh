#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
COMPARISON="${COMPARISON:-data/derived/models/model_family_comparison.json}"
OUT="${OUT:-data/derived/models/sequence_modality_diagnostic.json}"
CSV_OUT="${CSV_OUT:-data/derived/models/sequence_modality_diagnostic.csv}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli diagnose-sequence-comparison \
  --comparison "$COMPARISON" \
  --out "$OUT" \
  --csv-out "$CSV_OUT"
