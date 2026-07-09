#!/usr/bin/env bash
set -euo pipefail

INPUT="${INPUT:-data/derived/models/mountain_256x256_seeds40-42_20000.synthetic_event_list_targets_h6.csv}"
OUT="${OUT:-data/derived/models/synthetic_event_list_model/h6_event_list_model.json}"
PREDICTIONS_OUT="${PREDICTIONS_OUT:-data/derived/models/synthetic_event_list_model/h6_event_list_predictions.csv}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
SPLIT_FIELD="${SPLIT_FIELD:-}"
EPOCHS="${EPOCHS:-600}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
L2="${L2:-0.001}"
SEED="${SEED:-42}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli train-synthetic-event-list-model \
  --input "$INPUT" \
  --out "$OUT" \
  --predictions-out "$PREDICTIONS_OUT" \
  --train-fraction "$TRAIN_FRACTION" \
  --split-field "$SPLIT_FIELD" \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --l2 "$L2" \
  --seed "$SEED"
