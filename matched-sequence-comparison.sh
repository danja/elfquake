#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LOOKBACKS="${LOOKBACKS:-30 60 120}"
HIDDEN_UNITS_LIST="${HIDDEN_UNITS_LIST:-24}"
EPOCHS="${EPOCHS:-20}"
ROOT="${ROOT:-data/derived/models/sequence_sweep_20epoch}"

PYTHON_BIN="$PYTHON_BIN" \
  LOOKBACKS="$LOOKBACKS" \
  HIDDEN_UNITS_LIST="$HIDDEN_UNITS_LIST" \
  EPOCHS="$EPOCHS" \
  ROOT="$ROOT" \
  ./sweep-synthetic-sequence-model.sh

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli compare-model-run-summaries \
  --summary data/derived/models/mountain_256x256_seeds40-42_20000.sequence_model_run_summary.json \
  --summary "$ROOT/sequence_sweep_comparison.json" \
  --out "$ROOT/default_vs_matched_sequence_comparison.json" \
  --csv-out "$ROOT/default_vs_matched_sequence_comparison.csv"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli diagnose-sequence-comparison \
  --comparison "$ROOT/default_vs_matched_sequence_comparison.json" \
  --out "$ROOT/default_vs_matched_sequence_diagnostic.json" \
  --csv-out "$ROOT/default_vs_matched_sequence_diagnostic.csv"
