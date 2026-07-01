#!/usr/bin/env bash
set -euo pipefail

width="${WIDTH:-256}"
height="${HEIGHT:-256}"
steps="${STEPS:-100000}"
seed="${SEED:-42}"
snapshot_interval="${SNAPSHOT_INTERVAL:-10}"
progress_interval="${PROGRESS_INTERVAL:-100}"
bottom_layer_interval="${BOTTOM_LAYER_INTERVAL:-100}"
deposition_mode="${DEPOSITION_MODE:-sources}"
target_fill_limit="${TARGET_FILL_LIMIT:-$(( width * height / 16 ))}"
if [[ "$target_fill_limit" -lt 1 ]]; then
  target_fill_limit=1
fi
source_count="${SOURCE_COUNT:-$(( width * height / 64 ))}"
if [[ "$source_count" -lt 16 ]]; then
  source_count=16
fi
heatmap_scale="${HEATMAP_SCALE:-4}"
heatmap_color_min="${HEATMAP_COLOR_MIN:-0}"
heatmap_color_max="${HEATMAP_COLOR_MAX:-$width}"
heatmap_gamma="${HEATMAP_GAMMA:-0.85}"
heatmap_workers="${HEATMAP_WORKERS:-4}"
heatmap_progress_interval="${HEATMAP_PROGRESS_INTERVAL:-50}"
slope_threshold="${SLOPE_THRESHOLD:-$(( width / 16 ))}"
if [[ "$slope_threshold" -lt 4 ]]; then
  slope_threshold=4
fi
prefix="data/derived/sim/mountain_${width}x${height}_seed${seed}_${steps}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli run-sandpile-sim \
  --mountain-mode \
  --width "$width" --height "$height" --steps "$steps" \
  --threshold "$slope_threshold" \
  --deposition-mode "$deposition_mode" \
  --source-count "$source_count" --sensor-count 16 \
  --deposition-probability 0.7 --seed "$seed" \
  --target-fill-limit "$target_fill_limit" \
  --bottom-layer-removal-interval "$bottom_layer_interval" \
  --summary-out "${prefix}.summary.csv" \
  --sensors-out "${prefix}.sensors.csv" \
  --snapshot-dir "${prefix}.snapshots" \
  --snapshot-interval "$snapshot_interval" \
  --heatmap-dir "${prefix}.heatmaps" \
  --heatmap-scale "$heatmap_scale" \
  --heatmap-color-min "$heatmap_color_min" \
  --heatmap-color-max "$heatmap_color_max" \
  --heatmap-gamma "$heatmap_gamma" \
  --heatmap-workers "$heatmap_workers" \
  --heatmap-progress-interval "$heatmap_progress_interval" \
  --progress-interval "$progress_interval"
