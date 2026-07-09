#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/run-synthetic-diversity-smoke.sh

Generates extra synthetic seeds without heatmaps, video, or audio, then refreshes
event lists, aligned rows, tensors, and optional smoke evaluations for that seed set.

Environment:
  SEEDS             default "43 44"
  WIDTH             default 128
  HEIGHT            default 128
  STEPS             default 5000
  RUN_SIM           default 1
  RUN_ARTIFACTS     default 1
  RUN_EVALUATIONS   default 1
  RUN_EVENT_MAPS    default 0
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

seeds="${SEEDS:-43 44}"
width="${WIDTH:-128}"
height="${HEIGHT:-128}"
steps="${STEPS:-5000}"
run_sim="${RUN_SIM:-1}"
run_artifacts="${RUN_ARTIFACTS:-1}"

if [[ "$run_sim" != "0" ]]; then
  for seed in $seeds; do
    echo "synthetic diversity simulation: width=$width height=$height steps=$steps seed=$seed"
    WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" RUN_HEATMAPS=0 ./scripts/sim.sh
  done
else
  echo "simulation skipped"
fi

if [[ "$run_artifacts" != "0" ]]; then
  SEEDS="$seeds" WIDTH="$width" HEIGHT="$height" STEPS="$steps" \
    RUN_EVENT_MAPS="${RUN_EVENT_MAPS:-0}" RUN_EVALUATIONS="${RUN_EVALUATIONS:-1}" \
    ./scripts/refresh-synthetic-model-artifacts.sh
else
  echo "artifact refresh skipped"
fi
