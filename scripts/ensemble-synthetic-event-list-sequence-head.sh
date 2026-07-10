#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/synthetic_event_list_probes/h6/burn_0/targets.csv}"
ROOT="${ROOT:-data/derived/models/synthetic_event_list_sequence_ensemble}"
SEEDS="${SEEDS:-7 42 99}"
LOOKBACK_ROWS="${LOOKBACK_ROWS:-12}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-80}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
HIDDEN_UNITS="${HIDDEN_UNITS:-24}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
MAX_FEATURE_COUNT="${MAX_FEATURE_COUNT:-256}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
CALIBRATION_SOURCE="${CALIBRATION_SOURCE:-auto}"

if [[ ! -f "$INPUT" ]]; then
  echo "missing input target table: $INPUT" >&2
  exit 1
fi

mkdir -p "$ROOT"
report_args=()
for seed in $SEEDS; do
  run_dir="${ROOT}/seed_${seed}"
  mkdir -p "$run_dir"
  INPUT="$INPUT" \
  OUT="${run_dir}/sequence_head.json" \
  PREDICTIONS_OUT="${run_dir}/sequence_head_predictions.csv" \
  LOOKBACK_ROWS="$LOOKBACK_ROWS" \
  EPOCHS="$EPOCHS" \
  LEARNING_RATE="$LEARNING_RATE" \
  HIDDEN_UNITS="$HIDDEN_UNITS" \
  BATCH_SIZE="$BATCH_SIZE" \
  DROPOUT="$DROPOUT" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  MAX_FEATURE_COUNT="$MAX_FEATURE_COUNT" \
  VALIDATION_FRACTION="$VALIDATION_FRACTION" \
  EARLY_STOPPING_PATIENCE="$EARLY_STOPPING_PATIENCE" \
  CALIBRATION_SOURCE="$CALIBRATION_SOURCE" \
  SEED="$seed" \
    ./scripts/train-synthetic-event-list-sequence-head.sh
  report_args+=(--report "${run_dir}/sequence_head.json")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli ensemble-synthetic-event-list-sequence-heads \
  "${report_args[@]}" \
  --out "${ROOT}/ensemble.json" \
  --predictions-out "${ROOT}/ensemble_predictions.csv"

echo "ensemble: ${ROOT}/ensemble.json"
