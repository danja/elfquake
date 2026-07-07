#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DIAGNOSTIC="${DIAGNOSTIC:-data/derived/models/sequence_sweep_20epoch/default_vs_matched_sequence_diagnostic.json}"
OUT="${OUT:-data/derived/models/sequence_sweep_20epoch/default_vs_matched_sequence_selection.json}"
CSV_OUT="${CSV_OUT:-data/derived/models/sequence_sweep_20epoch/default_vs_matched_sequence_selection.csv}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-sequence-selection \
  --diagnostic "$DIAGNOSTIC" \
  --out "$OUT" \
  --csv-out "$CSV_OUT"
