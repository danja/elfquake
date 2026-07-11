#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ROOT:-data/derived/models/piezo_event_lead_time}"
OUT="${OUT:-$ROOT/analysis.json}"
PROFILE_OUT="${PROFILE_OUT:-$ROOT/profile.csv}"
SEEDS="${SEEDS:-10000 10001 10002 10100 10101 10102 10200 10201 10202}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-3000}"
EVENT_SUFFIX="${EVENT_SUFFIX:-}"

mkdir -p "$ROOT"
input_args=()
for seed in $SEEDS; do
  prefix="data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}"
  input_args+=(--piezo "${prefix}.piezo.csv")
  input_args+=(--events "${prefix}.avalanche_events${EVENT_SUFFIX}.csv")
done

lag_args=()
if [[ -n "${LAG_EDGES:-}" ]]; then
  for edge in $LAG_EDGES; do
    lag_args+=(--lag-edge "$edge")
  done
fi

signal_args=()
if [[ -n "${SIGNAL_FIELDS:-}" ]]; then
  for field in $SIGNAL_FIELDS; do
    signal_args+=(--signal-field "$field")
  done
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli analyze-piezo-event-lead-time \
  "${input_args[@]}" \
  "${lag_args[@]}" \
  "${signal_args[@]}" \
  --out "$OUT" \
  --profile-out "$PROFILE_OUT" \
  --primary-field "${PRIMARY_FIELD:-piezo_signal}" \
  --sensor-mode "${SENSOR_MODE:-mean}" \
  --sensor-top-k "${SENSOR_TOP_K:-3}" \
  --control-multiplier "${CONTROL_MULTIPLIER:-10}"
