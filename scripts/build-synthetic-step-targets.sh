#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SEEDS="${SEEDS:-12300 12301 12302 12400 12401 12402 12500 12501 12502}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-3000}"
HORIZON_STEPS="${HORIZON_STEPS:-15}"
STRIDE_STEPS="${STRIDE_STEPS:-5}"
ROOT="${ROOT:-data/derived/models/damage_short_horizon}"

args=()
for seed in $SEEDS; do
  prefix="data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}"
  args+=(--piezo "${prefix}.piezo.csv" --events "${prefix}.avalanche_events.csv")
done

mkdir -p "$ROOT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-synthetic-step-targets \
  "${args[@]}" \
  --out "$ROOT/targets.csv" \
  --report "$ROOT/targets.json" \
  --horizon-steps "$HORIZON_STEPS" \
  --stride-steps "$STRIDE_STEPS"
