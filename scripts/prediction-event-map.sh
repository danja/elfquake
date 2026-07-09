#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SEED="${SEED:-42}"
EVENTS="${EVENTS:-data/derived/sim/mountain_256x256_seed${SEED}_20000.avalanche_events.csv}"
WINDOWS="${WINDOWS:-data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv}"
REPORT="${REPORT:-data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.torch_tabular_group_seed${SEED}.json}"
OUT="${OUT:-data/derived/maps/mountain_256x256_seed${SEED}_20000.actual_vs_torch_predicted.png}"
METADATA="${METADATA:-${OUT%.png}.json}"
TITLE="${TITLE:-ELFQuake synthetic actual vs PyTorch predicted events}"

args=(
  -m elfquake.cli render-prediction-event-map
  --events "$EVENTS"
  --windows "$WINDOWS"
  --report "$REPORT"
  --out "$OUT"
  --metadata-out "$METADATA"
  --title "$TITLE"
)

if [[ "${EVALUATION:-}" ]]; then
  args+=(--evaluation "$EVALUATION")
fi
if [[ "${THRESHOLD:-}" ]]; then
  args+=(--threshold "$THRESHOLD")
fi
if [[ "${MAX_ACTUAL_EVENTS:-}" ]]; then
  args+=(--max-actual-events "$MAX_ACTUAL_EVENTS")
fi
if [[ "${BASEMAP_GEOJSON:-}" ]]; then
  args+=(--basemap-geojson "$BASEMAP_GEOJSON")
fi

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" "${args[@]}"
