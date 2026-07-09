#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/synthetic_event_list_probes/h6/burn_0/targets.csv}"
OUT_DIR="${OUT_DIR:-data/derived/models/synthetic_event_list_lagged_context}"
LAGS="${LAGS:-1 2 3 6}"
EPOCHS="${EPOCHS:-600}"
MAX_FEATURE_COUNT="${MAX_FEATURE_COUNT:-128}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$INPUT" ]]; then
  echo "missing input target table: $INPUT" >&2
  echo "run ./scripts/probe-synthetic-event-list-models.sh first, or set INPUT to an event-list target CSV" >&2
  exit 1
fi

lag_args=()
for lag in $LAGS; do
  lag_args+=(--lag "$lag")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-synthetic-lagged-context \
  --input "$INPUT" \
  --out "${OUT_DIR}/h6_lagged_targets.csv" \
  --report "${OUT_DIR}/h6_lagged_targets.json" \
  "${lag_args[@]}"

INPUT="${OUT_DIR}/h6_lagged_targets.csv" \
OUT="${OUT_DIR}/h6_lagged_temporal_model.json" \
PREDICTIONS_OUT="${OUT_DIR}/h6_lagged_temporal_predictions.csv" \
EPOCHS="$EPOCHS" \
MAX_FEATURE_COUNT="$MAX_FEATURE_COUNT" \
  ./scripts/train-synthetic-event-list-model.sh
