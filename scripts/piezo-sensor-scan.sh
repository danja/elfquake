#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/piezo-sensor-scan.sh [piezo_csv] [output_csv]

Defaults:
  piezo_csv  data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${SEED}_${STEPS}.piezo.csv
  output_csv same prefix with .piezo_sensor_scan.csv

Environment:
  WIDTH                    default 256
  HEIGHT                   default 256
  STEPS                    default 10000
  SEED                     default 42
  VLF_IMAGE_ROOT           default data/raw/vlf/cumiana/captures
  VLF_FILENAME_PREFIX      default last_E_VLF
  SIM_STEP_SECONDS         default 60
  VLF_COLUMN_SECONDS       default 1
  SENSOR_ID                optional single sensor id to scan
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

vlf_image_root="${VLF_IMAGE_ROOT:-data/raw/vlf/cumiana/captures}"
if [[ ! -d "$vlf_image_root" ]]; then
  echo "error: VLF image root not found: $vlf_image_root" >&2
  echo "Capture Cumiana images first, or set VLF_IMAGE_ROOT." >&2
  exit 2
fi

prefix="${input%.piezo.csv}"
output="${2:-${prefix}.piezo_sensor_scan.csv}"
vlf_filename_prefix="${VLF_FILENAME_PREFIX:-last_E_VLF}"
sim_step_seconds="${SIM_STEP_SECONDS:-60}"
vlf_column_seconds="${VLF_COLUMN_SECONDS:-1}"

args=(
  -m elfquake.cli scan-piezo-sensors
  --real-vlf-image-root "$vlf_image_root"
  --real-vlf-filename-prefix "$vlf_filename_prefix"
  --sim-piezo "$input"
  --sim-step-seconds "$sim_step_seconds"
  --vlf-column-seconds "$vlf_column_seconds"
  --out "$output"
)

if [[ "${SENSOR_ID:-}" ]]; then
  args+=(--sensor-id "$SENSOR_ID")
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python "${args[@]}"
