#!/usr/bin/env bash
set -euo pipefail

width="${WIDTH:-256}"
height="${HEIGHT:-256}"
steps="${STEPS:-10000}"
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
run_heatmaps="${RUN_HEATMAPS:-1}"
piezo_sensor_count="${PIEZO_SENSOR_COUNT:-16}"
piezo_cluster_count="${PIEZO_CLUSTER_COUNT:-8}"
piezo_activation_ratio="${PIEZO_ACTIVATION_RATIO:-0.75}"
piezo_susceptibility_base="${PIEZO_SUSCEPTIBILITY_BASE:-0.15}"
piezo_susceptibility_variation="${PIEZO_SUSCEPTIBILITY_VARIATION:-0.85}"
piezo_cluster_radius="${PIEZO_CLUSTER_RADIUS:-0}"
piezo_attenuation_radius="${PIEZO_ATTENUATION_RADIUS:-16}"
piezo_max_distance_radius="${PIEZO_MAX_DISTANCE_RADIUS:-48}"
piezo_charge_decay="${PIEZO_CHARGE_DECAY:-0.995}"
piezo_charge_coupling="${PIEZO_CHARGE_COUPLING:-1.0}"
piezo_release_charge_threshold="${PIEZO_RELEASE_CHARGE_THRESHOLD:-40}"
piezo_release_ratio="${PIEZO_RELEASE_RATIO:-0.25}"
piezo_critical_release_ratio="${PIEZO_CRITICAL_RELEASE_RATIO:-0.10}"
piezo_saturation="${PIEZO_SATURATION:-1000}"
avalanche_signal_attenuation_radius="${AVALANCHE_SIGNAL_ATTENUATION_RADIUS:-0}"
avalanche_signal_max_distance_radius="${AVALANCHE_SIGNAL_MAX_DISTANCE_RADIUS:-0}"
slope_threshold="${SLOPE_THRESHOLD:-$(( width / 16 ))}"
if [[ "$slope_threshold" -lt 4 ]]; then
  slope_threshold=4
fi
prefix="data/derived/sim/mountain_${width}x${height}_seed${seed}_${steps}"

args=(
  -m elfquake.cli run-sandpile-sim
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
  --piezo-out "${prefix}.piezo.csv" \
  --avalanche-signal-out "${prefix}.avalanche_signal.csv" \
  --avalanche-activity-out "${prefix}.avalanche_activity.csv" \
  --piezo-sensor-count "$piezo_sensor_count" \
  --piezo-cluster-count "$piezo_cluster_count" \
  --piezo-activation-ratio "$piezo_activation_ratio" \
  --piezo-susceptibility-base "$piezo_susceptibility_base" \
  --piezo-susceptibility-variation "$piezo_susceptibility_variation" \
  --piezo-cluster-radius "$piezo_cluster_radius" \
  --piezo-attenuation-radius "$piezo_attenuation_radius" \
  --piezo-max-distance-radius "$piezo_max_distance_radius" \
  --piezo-charge-decay "$piezo_charge_decay" \
  --piezo-charge-coupling "$piezo_charge_coupling" \
  --piezo-release-charge-threshold "$piezo_release_charge_threshold" \
  --piezo-release-ratio "$piezo_release_ratio" \
  --piezo-critical-release-ratio "$piezo_critical_release_ratio" \
  --piezo-saturation "$piezo_saturation" \
  --avalanche-signal-attenuation-radius "$avalanche_signal_attenuation_radius" \
  --avalanche-signal-max-distance-radius "$avalanche_signal_max_distance_radius"
)

if [[ "$run_heatmaps" != "0" ]]; then
  args+=(
  --snapshot-dir "${prefix}.snapshots" \
  --snapshot-interval "$snapshot_interval" \
  --heatmap-dir "${prefix}.heatmaps" \
  --heatmap-scale "$heatmap_scale" \
  --heatmap-color-min "$heatmap_color_min" \
  --heatmap-color-max "$heatmap_color_max" \
  --heatmap-gamma "$heatmap_gamma" \
  --heatmap-workers "$heatmap_workers" \
  --heatmap-progress-interval "$heatmap_progress_interval"
  )
fi

args+=(--progress-interval "$progress_interval")

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python "${args[@]}"
