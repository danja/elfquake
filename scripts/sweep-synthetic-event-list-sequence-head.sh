#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/synthetic_event_list_probes/h6/burn_0/targets.csv}"
ROOT="${ROOT:-data/derived/models/synthetic_event_list_sequence_sweep}"
SEEDS="${SEEDS:-7 42 99}"
LOOKBACK_ROWS_LIST="${LOOKBACK_ROWS_LIST:-8 12}"
DROPOUTS="${DROPOUTS:-0.1 0.2 0.3}"
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
  echo "run ./scripts/probe-synthetic-event-list-models.sh first, or set INPUT to an event-list target CSV" >&2
  exit 1
fi

mkdir -p "$ROOT"

slug_number() {
  local value="$1"
  value="${value//./p}"
  echo "$value"
}

for lookback in $LOOKBACK_ROWS_LIST; do
  for dropout in $DROPOUTS; do
    dropout_slug="$(slug_number "$dropout")"
    for seed in $SEEDS; do
      run_dir="${ROOT}/lookback_${lookback}/dropout_${dropout_slug}/seed_${seed}"
      mkdir -p "$run_dir"
      INPUT="$INPUT" \
      OUT="${run_dir}/sequence_head.json" \
      PREDICTIONS_OUT="${run_dir}/sequence_head_predictions.csv" \
      LOOKBACK_ROWS="$lookback" \
      EPOCHS="$EPOCHS" \
      LEARNING_RATE="$LEARNING_RATE" \
      HIDDEN_UNITS="$HIDDEN_UNITS" \
      BATCH_SIZE="$BATCH_SIZE" \
      DROPOUT="$dropout" \
      WEIGHT_DECAY="$WEIGHT_DECAY" \
      MAX_FEATURE_COUNT="$MAX_FEATURE_COUNT" \
      VALIDATION_FRACTION="$VALIDATION_FRACTION" \
      EARLY_STOPPING_PATIENCE="$EARLY_STOPPING_PATIENCE" \
      CALIBRATION_SOURCE="$CALIBRATION_SOURCE" \
      SEED="$seed" \
        ./scripts/train-synthetic-event-list-sequence-head.sh
    done
  done
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-synthetic-event-list-sequence-heads \
  --root "$ROOT" \
  --out "${ROOT}/summary.json" \
  --csv-out "${ROOT}/summary.csv"

echo "summary: ${ROOT}/summary.csv"
