#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./compare-piezo-vlf.sh [sim_png] [output_csv]

Defaults:
  sim_png    data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${SEED}_${STEPS}.piezo_vlf_summary.png
  output_csv same prefix with .piezo_vlf_comparison.csv

Environment:
  WIDTH           default 256
  HEIGHT          default 256
  STEPS           default 10000
  SEED            default 42
  REAL_IMAGE_ROOT default data/raw/vlf/cumiana/captures
  FILENAME_PREFIX default last_E_VLF
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

width="${WIDTH:-256}"
height="${HEIGHT:-256}"
steps="${STEPS:-10000}"
seed="${SEED:-42}"
prefix="data/derived/sim/mountain_${width}x${height}_seed${seed}_${steps}"
sim_png="${1:-${prefix}.piezo_vlf_summary.png}"
output="${2:-${prefix}.piezo_vlf_comparison.csv}"
real_root="${REAL_IMAGE_ROOT:-data/raw/vlf/cumiana/captures}"
filename_prefix="${FILENAME_PREFIX:-last_E_VLF}"

if [[ ! -f "$sim_png" ]]; then
  echo "error: simulated VLF PNG not found: $sim_png" >&2
  echo "Run ./run-all.sh first, or pass a simulated PNG explicitly." >&2
  exit 2
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli compare-vlf-image-features \
  --sim-image "$sim_png" \
  --real-image-root "$real_root" \
  --filename-prefix "$filename_prefix" \
  --out "$output"
