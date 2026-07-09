#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/synthetic_event_list_probes/h6/burn_0/targets.csv}"
OUT="${OUT:-data/derived/models/synthetic_event_list_sequence/h6_sequence_head.json}"
PREDICTIONS_OUT="${PREDICTIONS_OUT:-data/derived/models/synthetic_event_list_sequence/h6_sequence_head_predictions.csv}"
LOOKBACK_ROWS="${LOOKBACK_ROWS:-12}"
EPOCHS="${EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
HIDDEN_UNITS="${HIDDEN_UNITS:-24}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DROPOUT="${DROPOUT:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
MAX_FEATURE_COUNT="${MAX_FEATURE_COUNT:-256}"
SEED="${SEED:-42}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli train-synthetic-event-list-sequence-head \
  --input "$INPUT" \
  --out "$OUT" \
  --predictions-out "$PREDICTIONS_OUT" \
  --lookback-rows "$LOOKBACK_ROWS" \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --hidden-units "$HIDDEN_UNITS" \
  --batch-size "$BATCH_SIZE" \
  --dropout "$DROPOUT" \
  --weight-decay "$WEIGHT_DECAY" \
  --max-feature-count "$MAX_FEATURE_COUNT" \
  --seed "$SEED"
