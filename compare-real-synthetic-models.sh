#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REAL_REPORT="${REAL_REPORT:-data/derived/models/central_italy.ingv_backfill_seismic_windows.temporal_holdout.json}"
REAL_SUMMARY="${REAL_SUMMARY:-data/derived/models/central_italy.ingv_backfill_seismic_windows.model_run_summary.json}"
SYNTHETIC_SEQUENCE_SUMMARY="${SYNTHETIC_SEQUENCE_SUMMARY:-data/derived/models/mountain_256x256_seeds40-42_20000.sequence_model_run_summary.json}"
REGIME_SUMMARY="${REGIME_SUMMARY:-data/derived/models/sequence_full_regime/sequence_full_model_run_summary.json}"
OUT="${OUT:-data/derived/models/real_synthetic_compact_comparison.json}"
CSV_OUT="${CSV_OUT:-data/derived/models/real_synthetic_compact_comparison.csv}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-run-reports \
  --report "$REAL_REPORT" \
  --out "$REAL_SUMMARY"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli compare-model-run-summaries \
  --summary "$REAL_SUMMARY" \
  --summary "$SYNTHETIC_SEQUENCE_SUMMARY" \
  --summary "$REGIME_SUMMARY" \
  --out "$OUT" \
  --csv-out "$CSV_OUT"
