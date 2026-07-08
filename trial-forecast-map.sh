#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EVENTS="${EVENTS:-data/derived/models/trial_forecast/mag_gt2_weekly_trial_events.csv}"
OUT="${OUT:-data/derived/maps/mag_gt2_weekly_trial_forecast_map.png}"
METADATA="${METADATA:-${OUT%.png}.json}"
TITLE="${TITLE:-ELFQuake trial forecast event coordinates - not validated}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli render-event-map \
  --events "$EVENTS" \
  --out "$OUT" \
  --metadata-out "$METADATA" \
  --title "$TITLE"
