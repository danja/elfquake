#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./piezo-audio.sh [piezo_csv] [output_wav]

Defaults:
  piezo_csv  newest data/derived/sim/*.piezo.csv
  output_wav same prefix with .piezo.wav

Environment:
  SAMPLE_RATE       default 44100
  DURATION_SECONDS  default 20
  GAIN              default 0.95
  SMOOTH_STEPS      default 64
  SENSOR_ID         default 0
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
  input="$(find data/derived/sim -maxdepth 1 -type f -name '*.piezo.csv' -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1s/^[^ ]* //p')"
fi

if [[ -z "$input" || ! -f "$input" ]]; then
  echo "error: no piezo CSV found. Run ./sim.sh, or pass a .piezo.csv path explicitly." >&2
  exit 2
fi

prefix="${input%.piezo.csv}"
output="${2:-${prefix}.piezo.wav}"
sample_rate="${SAMPLE_RATE:-44100}"
duration_seconds="${DURATION_SECONDS:-20}"
gain="${GAIN:-0.95}"
smooth_steps="${SMOOTH_STEPS:-64}"
sensor_id="${SENSOR_ID:-0}"
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
