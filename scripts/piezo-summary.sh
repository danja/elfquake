#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/piezo-summary.sh [piezo_csv] [output_png] [metadata_json]

Defaults:
  piezo_csv     data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${SEED}_${STEPS}.piezo.csv
  output_png    same prefix with .piezo_summary.png
  metadata_json same prefix with .piezo_summary.json

Environment:
  WIDTH              default 256
  HEIGHT             default 256
  STEPS              default 10000
  SEED               default 42
  STEP_SECONDS       default 60
  FREQ_BINS          default 96
  WINDOW_STEPS       default 64
  SCALE              default 4
  TIMESERIES_HEIGHT  default 48
  OUTPUT_WIDTH       default 1600
  SENSOR_ID          default 5
  DC_BLOCK           default 0.995
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" ]]; then
  input="$1"
else
  width="${WIDTH:-256}"
  height="${HEIGHT:-256}"
  steps="${STEPS:-10000}"
  seed="${SEED:-42}"
  input="data/derived/sim/mountain_${width}x${height}_seed${seed}_${steps}.piezo.csv"
fi

if [[ -z "$input" || ! -f "$input" ]]; then
  echo "error: piezo CSV not found: $input" >&2
  echo "Run ./scripts/sim.sh with matching WIDTH/HEIGHT/STEPS/SEED, or pass a .piezo.csv path explicitly." >&2
  exit 2
fi

prefix="${input%.piezo.csv}"
output="${2:-${prefix}.piezo_summary.png}"
metadata="${3:-${prefix}.piezo_summary.json}"
step_seconds="${STEP_SECONDS:-60}"
freq_bins="${FREQ_BINS:-96}"
window_steps="${WINDOW_STEPS:-64}"
scale="${SCALE:-4}"
timeseries_height="${TIMESERIES_HEIGHT:-48}"
output_width="${OUTPUT_WIDTH:-1600}"
sensor_id="${SENSOR_ID:-5}"
dc_block="${DC_BLOCK:-0.995}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli render-piezo-summary \
  --piezo "$input" \
  --out "$output" \
  --metadata-out "$metadata" \
  --step-seconds "$step_seconds" \
  --freq-bins "$freq_bins" \
  --window-steps "$window_steps" \
  --scale "$scale" \
  --timeseries-height "$timeseries_height" \
  --output-width "$output_width" \
  --sensor-id "$sensor_id" \
  --dc-block "$dc_block"
