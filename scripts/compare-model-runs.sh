#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TABULAR_SUMMARY="${TABULAR_SUMMARY:-data/derived/models/mountain_256x256_seeds40-42_20000.model_run_summary.json}"
SEQUENCE_SUMMARY="${SEQUENCE_SUMMARY:-data/derived/models/mountain_256x256_seeds40-42_20000.sequence_model_run_summary.json}"
OUT="${OUT:-data/derived/models/mountain_256x256_seeds40-42_20000.tabular_vs_sequence_model_comparison.json}"
CSV_OUT="${CSV_OUT:-data/derived/models/mountain_256x256_seeds40-42_20000.tabular_vs_sequence_model_comparison.csv}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli compare-model-run-summaries \
  --summary "$TABULAR_SUMMARY" \
  --summary "$SEQUENCE_SUMMARY" \
  --out "$OUT" \
  --csv-out "$CSV_OUT"
