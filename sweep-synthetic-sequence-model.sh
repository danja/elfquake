#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LOOKBACKS="${LOOKBACKS:-30 60 120}"
HIDDEN_UNITS_LIST="${HIDDEN_UNITS_LIST:-24}"
EPOCHS="${EPOCHS:-10}"
ROOT="${ROOT:-data/derived/models/sequence_sweep}"
SUMMARY_ARGS=()

for LOOKBACK in $LOOKBACKS; do
  for HIDDEN_UNITS in $HIDDEN_UNITS_LIST; do
    RUN_DIR="$ROOT/lookback_${LOOKBACK}_hidden_${HIDDEN_UNITS}"
    mkdir -p "$RUN_DIR"
    LOOKBACK_STEPS="$LOOKBACK" \
      HIDDEN_UNITS="$HIDDEN_UNITS" \
      EPOCHS="$EPOCHS" \
      PYTHON_BIN="$PYTHON_BIN" \
      OUT="$RUN_DIR/torch_sequence.json" \
      SUMMARY="$RUN_DIR/sequence_model_run_summary.json" \
      GROUP_PREFIX="$RUN_DIR/torch_sequence_group" \
      ./train-synthetic-sequence-model.sh
    SUMMARY_ARGS+=(--summary "$RUN_DIR/sequence_model_run_summary.json")
  done
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli compare-model-run-summaries \
  "${SUMMARY_ARGS[@]}" \
  --out "$ROOT/sequence_sweep_comparison.json" \
  --csv-out "$ROOT/sequence_sweep_comparison.csv"
