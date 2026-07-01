#!/usr/bin/env bash
set -euo pipefail

width="${WIDTH:-256}"
height="${HEIGHT:-256}"
steps="${STEPS:-10000}"
seed="${SEED:-42}"
snapshot_interval="${SNAPSHOT_INTERVAL:-100}"
bottom_layer_interval="${BOTTOM_LAYER_INTERVAL:-100}"
heatmap_scale="${HEATMAP_SCALE:-4}"
heatmap_color_max="${HEATMAP_COLOR_MAX:-$width}"
slope_threshold="${SLOPE_THRESHOLD:-$(( width / 16 ))}"
if [[ "$slope_threshold" -lt 4 ]]; then
  slope_threshold=4
fi
prefix="data/derived/sim/mountain_${width}x${height}_seed${seed}_${steps}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli run-sandpile-sim \
  --mountain-mode \
  --width "$width" --height "$height" --steps "$steps" \
  --threshold "$slope_threshold" \
  --source-count 16 --sensor-count 16 \
  --deposition-probability 0.7 --seed "$seed" \
  --bottom-layer-removal-interval "$bottom_layer_interval" \
  --summary-out "${prefix}.summary.csv" \
  --sensors-out "${prefix}.sensors.csv" \
  --snapshot-dir "${prefix}.snapshots" \
  --snapshot-interval "$snapshot_interval" \
  --heatmap-dir "${prefix}.heatmaps" \
  --heatmap-scale "$heatmap_scale" \
  --heatmap-color-max "$heatmap_color_max" \
  --progress-interval "$snapshot_interval"
