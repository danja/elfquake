#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TRAINING_SEEDS="${TRAINING_SEEDS:-40 41 42}"
ROOT="${ROOT:-data/derived/models/sequence_training_seed_repeat}"
LOOKBACK_STEPS="${LOOKBACK_STEPS:-60}"
HIDDEN_UNITS="${HIDDEN_UNITS:-24}"
EPOCHS="${EPOCHS:-20}"
SUMMARY_ARGS=()

for TRAINING_SEED in $TRAINING_SEEDS; do
  RUN_DIR="$ROOT/torch_seed_${TRAINING_SEED}"
  mkdir -p "$RUN_DIR"
  PYTHON_BIN="$PYTHON_BIN" \
    SEED="$TRAINING_SEED" \
    LOOKBACK_STEPS="$LOOKBACK_STEPS" \
    HIDDEN_UNITS="$HIDDEN_UNITS" \
    EPOCHS="$EPOCHS" \
    OUT="$RUN_DIR/torch_sequence.json" \
    SUMMARY="$RUN_DIR/sequence_model_run_summary.json" \
    GROUP_PREFIX="$RUN_DIR/torch_sequence_group" \
    ./train-synthetic-sequence-model.sh
  SUMMARY_ARGS+=(--summary "$RUN_DIR/sequence_model_run_summary.json")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli compare-model-run-summaries \
  "${SUMMARY_ARGS[@]}" \
  --out "$ROOT/sequence_training_seed_comparison.json" \
  --csv-out "$ROOT/sequence_training_seed_comparison.csv"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli diagnose-sequence-comparison \
  --comparison "$ROOT/sequence_training_seed_comparison.json" \
  --out "$ROOT/sequence_training_seed_diagnostic.json" \
  --csv-out "$ROOT/sequence_training_seed_diagnostic.csv"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-sequence-selection \
  --diagnostic "$ROOT/sequence_training_seed_diagnostic.json" \
  --out "$ROOT/sequence_training_seed_selection.json" \
  --csv-out "$ROOT/sequence_training_seed_selection.csv"
