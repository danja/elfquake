#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/run-all.sh

Runs the simulation demo pipeline in dependency order:
  1. sim.sh
  2. direct seismic synthetic event list
  3. piezo-vlf-summary.sh
  4. optional piezo-sensor-scan.sh
  5. piezo-audio.sh
  6. make-video.sh
  7. event-map.sh

Defaults match sim.sh:
  WIDTH=256 HEIGHT=256 STEPS=10000 SEED=42

Environment:
  RUN_SIM=0        skip sim.sh and reuse existing outputs
  RUN_HEATMAPS=0   skip snapshot and heatmap PNG generation inside sim.sh
  RUN_VIDEO=0      skip make-video.sh
  RUN_AUDIO=0      skip piezo WAV sonification
  RUN_EVENT_MAP=0  skip direct seismic event list and event-map.sh
  RUN_SENSOR_SCAN=1 rank piezo sensors against local Cumiana VLF captures
  RUN_FFT=1        also render the FFT diagnostic PNG
  AVALANCHE_EVENT_QUANTILE default 0.99
  AVALANCHE_EVENT_WINDOW   default 30
  AVALANCHE_EVENT_MAX      default 0, no cap
  AVALANCHE_SPATIAL_PROFILE default italy_apennines
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
run_heatmaps="${RUN_HEATMAPS:-1}"
run_video="${RUN_VIDEO:-1}"
run_audio="${RUN_AUDIO:-1}"
run_event_map="${RUN_EVENT_MAP:-1}"
run_sensor_scan="${RUN_SENSOR_SCAN:-0}"
run_fft="${RUN_FFT:-0}"
fps="${FPS:-20}"
avalanche_event_quantile="${AVALANCHE_EVENT_QUANTILE:-0.99}"
avalanche_event_window="${AVALANCHE_EVENT_WINDOW:-30}"
avalanche_event_max="${AVALANCHE_EVENT_MAX:-0}"
avalanche_spatial_profile="${AVALANCHE_SPATIAL_PROFILE:-italy_apennines}"

echo "prefix: $prefix"

if [[ "$run_sim" != "0" ]]; then
  echo "step 1/7: simulation"
  WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" RUN_HEATMAPS="$run_heatmaps" ./scripts/sim.sh
else
  echo "step 1/7: simulation skipped"
fi

if [[ "$run_event_map" != "0" ]]; then
  echo "step 2/7: direct seismic synthetic event list"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli build-synthetic-event-list \
    --summary "${prefix}.summary.csv" \
    --sensors "${prefix}.sensors.csv" \
    --grid-width "$width" \
    --grid-height "$height" \
    --out "${prefix}.synthetic_events.csv"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli build-avalanche-signal-event-list \
    --avalanche "${prefix}.avalanche_signal.csv" \
    --activity "${prefix}.avalanche_activity.csv" \
    --grid-width "$width" \
    --grid-height "$height" \
    --min-signal-quantile "$avalanche_event_quantile" \
    --local-max-window "$avalanche_event_window" \
    --max-events "$avalanche_event_max" \
    --spatial-profile "$avalanche_spatial_profile" \
    --out "${prefix}.avalanche_events.csv"
else
  echo "step 2/7: direct seismic synthetic event list skipped"
fi

if [[ "$run_fft" != "0" ]]; then
  echo "optional: piezo FFT diagnostic summary"
  WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" ./scripts/piezo-summary.sh "${prefix}.piezo.csv"
fi

echo "step 3/7: piezo VLF summary"
WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" ./scripts/piezo-vlf-summary.sh "${prefix}.piezo.csv"

if [[ "$run_sensor_scan" != "0" ]]; then
  echo "step 4/7: piezo sensor scan"
  WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" ./scripts/piezo-sensor-scan.sh "${prefix}.piezo.csv"
else
  echo "step 4/7: piezo sensor scan skipped"
fi

if [[ "$run_audio" != "0" ]]; then
  echo "step 5/7: piezo WAV sonification"
  WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" ./scripts/piezo-audio.sh "${prefix}.piezo.csv"
else
  echo "step 5/7: piezo WAV sonification skipped"
fi

if [[ "$run_video" != "0" && "$run_heatmaps" == "0" ]]; then
  echo "error: RUN_VIDEO=1 requires RUN_HEATMAPS=1 so heatmap frames exist" >&2
  exit 2
fi

if [[ "$run_video" != "0" ]]; then
  echo "step 6/7: heatmap video"
  WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" ./scripts/make-video.sh "${prefix}.heatmaps" "${prefix}.mp4" "$fps"
else
  echo "step 6/7: heatmap video skipped"
fi

if [[ "$run_event_map" != "0" ]]; then
  echo "step 7/7: synthetic event map"
  ./scripts/event-map.sh "${prefix}.avalanche_events.csv"
else
  echo "step 7/7: synthetic event map skipped"
fi

echo "done: $prefix"
