#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./piezo-audio.sh [piezo_csv] [output_wav]

Defaults:
  piezo_csv  data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${SEED}_${STEPS}.piezo.csv
  output_wav same prefix with .piezo.wav

Environment:
  WIDTH            default 256
  HEIGHT           default 256
  STEPS            default 10000
  SEED             default 42
  SAMPLE_RATE       default 44100
  DURATION_SECONDS  default 20
  GAIN              default 0.95
  SMOOTH_STEPS      default 64
  SENSOR_ID         default 5
  DC_BLOCK          default 0.995

The WAV is a time-compressed sonification of the summed piezo signal, not a
physical radio waveform.
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
  echo "Run ./sim.sh with matching WIDTH/HEIGHT/STEPS/SEED, or pass a .piezo.csv path explicitly." >&2
  exit 2
fi

prefix="${input%.piezo.csv}"
if [[ "$prefix" == "$input" ]]; then
  prefix="${input%.avalanche_signal.csv}"
fi
output="${2:-${prefix}.piezo.wav}"
sample_rate="${SAMPLE_RATE:-44100}"
duration_seconds="${DURATION_SECONDS:-20}"
gain="${GAIN:-0.95}"
smooth_steps="${SMOOTH_STEPS:-64}"
sensor_id="${SENSOR_ID:-5}"
dc_block="${DC_BLOCK:-0.995}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli render-piezo-audio \
  --piezo "$input" \
  --out "$output" \
  --sample-rate "$sample_rate" \
  --duration-seconds "$duration_seconds" \
  --gain "$gain" \
  --smooth-steps "$smooth_steps" \
  --sensor-id "$sensor_id" \
  --dc-block "$dc_block"
