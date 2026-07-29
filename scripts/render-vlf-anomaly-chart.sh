#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SCORES="${SCORES:-data/derived/models/self_supervised/real_vlf_anomaly_scores.csv}"
OUT="${OUT:-docs/images/anomaly.png}"

if [[ ! -f "$SCORES" ]]; then
  echo "error: anomaly scores not found: $SCORES" >&2
  exit 2
fi

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli render-vlf-anomaly-chart \
  --scores "$SCORES" \
  --out "$OUT" \
  --alert-threshold "${ALERT_THRESHOLD:-0.8}" \
  --max-gap-hours "${MAX_GAP_HOURS:-1.0}"
