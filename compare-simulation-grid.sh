#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./compare-simulation-grid.sh

Runs the simulation/report pipeline across multiple seeds with the same comparison reports.

Environment:
  SEEDS             default "40 41 42"
  WIDTH             default 128
  HEIGHT            default 128
  STEPS             default 2000
  RUN_SIM           default 1
  RUN_HEATMAPS      default 0
  RUN_VIDEO         default 0
  RUN_AUDIO         default 0
  REAL_EVENTS       optional normalized INGV-like CSV for shape comparison
  REAL_IMAGE_ROOT   default data/raw/vlf/cumiana/captures
  AVALANCHE_EVENT_QUANTILE default 0.99
  AVALANCHE_EVENT_WINDOW   default 30
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

seeds="${SEEDS:-40 41 42}"
width="${WIDTH:-128}"
height="${HEIGHT:-128}"
steps="${STEPS:-2000}"
run_sim="${RUN_SIM:-1}"
run_heatmaps="${RUN_HEATMAPS:-0}"
run_video="${RUN_VIDEO:-0}"
run_audio="${RUN_AUDIO:-0}"
real_events="${REAL_EVENTS:-}"
real_image_root="${REAL_IMAGE_ROOT:-data/raw/vlf/cumiana/captures}"
avalanche_event_quantile="${AVALANCHE_EVENT_QUANTILE:-0.99}"
avalanche_event_window="${AVALANCHE_EVENT_WINDOW:-30}"

for seed in $seeds; do
  prefix="data/derived/sim/mountain_${width}x${height}_seed${seed}_${steps}"
  echo "simulation grid item: $prefix"
  WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" \
    RUN_SIM="$run_sim" RUN_HEATMAPS="$run_heatmaps" RUN_VIDEO="$run_video" RUN_AUDIO="$run_audio" \
    AVALANCHE_EVENT_QUANTILE="$avalanche_event_quantile" \
    AVALANCHE_EVENT_WINDOW="$avalanche_event_window" \
    ./run-all.sh

  if [[ -d "$real_image_root" ]]; then
    WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" \
      REAL_IMAGE_ROOT="$real_image_root" \
      ./compare-piezo-vlf.sh
  fi

  if [[ -n "$real_events" && -f "$real_events" ]]; then
    WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" \
      REAL_EVENTS="$real_events" REAL_VLF_ROOT="$real_image_root" \
      ./compare-signal-shapes.sh
  else
    WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" \
      REAL_VLF_ROOT="$real_image_root" \
      ./compare-signal-shapes.sh
  fi
done
