#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/piezo-vlf-summary.sh [piezo_csv] [output_png] [metadata_json]

Defaults:
  piezo_csv     data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${SEED}_${STEPS}.piezo.csv
  output_png    same prefix with .piezo_vlf_summary.png
  metadata_json same prefix with .piezo_vlf_summary.json

Environment:
  WIDTH                default 256
  HEIGHT               default 256
  STEPS                default 10000
  SEED                 default 42
  CARRIER_FREQ_MIN_HZ  default 0
  CARRIER_FREQ_MAX_HZ  default 24000
  FREQ_BINS            default 192
  SCALE                default 4
  TIMESERIES_HEIGHT    default 48
  OUTPUT_WIDTH         default 1600
  SENSOR_ID            default 5
  DC_BLOCK             default 0.995
  DISPLAY_COLOR_QUANTILE default 0.82
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
if [[ "$prefix" == "$input" ]]; then
  prefix="${input%.avalanche_signal.csv}"
fi
output="${2:-${prefix}.piezo_vlf_summary.png}"
metadata="${3:-${prefix}.piezo_vlf_summary.json}"
carrier_freq_min_hz="${CARRIER_FREQ_MIN_HZ:-0}"
carrier_freq_max_hz="${CARRIER_FREQ_MAX_HZ:-24000}"
freq_bins="${FREQ_BINS:-192}"
scale="${SCALE:-4}"
timeseries_height="${TIMESERIES_HEIGHT:-48}"
output_width="${OUTPUT_WIDTH:-1600}"
sensor_id="${SENSOR_ID:-5}"
dc_block="${DC_BLOCK:-0.995}"
display_color_quantile="${DISPLAY_COLOR_QUANTILE:-0.82}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli render-piezo-vlf-summary \
  --piezo "$input" \
  --out "$output" \
  --metadata-out "$metadata" \
  --carrier-freq-min-hz "$carrier_freq_min_hz" \
  --carrier-freq-max-hz "$carrier_freq_max_hz" \
  --freq-bins "$freq_bins" \
  --scale "$scale" \
  --timeseries-height "$timeseries_height" \
  --output-width "$output_width" \
  --sensor-id "$sensor_id" \
  --dc-block "$dc_block" \
  --display-color-quantile "$display_color_quantile"
