#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run-all.sh

Runs the simulation demo pipeline in dependency order:
  1. sim.sh
  2. synthetic event list
  3. piezo-summary.sh
  4. piezo-vlf-summary.sh
  5. piezo-audio.sh
  6. make-video.sh
  7. event-map.sh

Defaults match sim.sh:
  WIDTH=256 HEIGHT=256 STEPS=10000 SEED=42

Environment:
  RUN_SIM=0        skip sim.sh and reuse existing outputs
  RUN_VIDEO=0      skip make-video.sh
  RUN_EVENT_MAP=0  skip synthetic event list and event-map.sh
  FPS=20           video frame rate
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
run_sim="${RUN_SIM:-1}"
run_video="${RUN_VIDEO:-1}"
run_event_map="${RUN_EVENT_MAP:-1}"
fps="${FPS:-20}"

echo "prefix: $prefix"

if [[ "$run_sim" != "0" ]]; then
  echo "step 1/7: simulation"
  WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" ./sim.sh
else
  echo "step 1/7: simulation skipped"
fi

if [[ "$run_event_map" != "0" ]]; then
  echo "step 2/7: synthetic event list"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli build-synthetic-event-list \
    --summary "${prefix}.summary.csv" \
    --sensors "${prefix}.sensors.csv" \
    --grid-width "$width" \
    --grid-height "$height" \
    --out "${prefix}.synthetic_events.csv"
else
  echo "step 2/7: synthetic event list skipped"
fi

echo "step 3/7: piezo FFT diagnostic summary"
WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" ./piezo-summary.sh "${prefix}.piezo.csv"

echo "step 4/7: piezo VLF analogue summary"
WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" ./piezo-vlf-summary.sh "${prefix}.piezo.csv"

echo "step 5/7: piezo WAV sonification"
WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" ./piezo-audio.sh "${prefix}.piezo.csv"

if [[ "$run_video" != "0" ]]; then
  echo "step 6/7: heatmap video"
  WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" ./make-video.sh "${prefix}.heatmaps" "${prefix}.mp4" "$fps"
else
  echo "step 6/7: heatmap video skipped"
fi

if [[ "$run_event_map" != "0" ]]; then
  echo "step 7/7: synthetic event map"
  ./event-map.sh "${prefix}.synthetic_events.csv"
else
  echo "step 7/7: synthetic event map skipped"
fi

echo "done: $prefix"
