#!/usr/bin/env bash
set -euo pipefail

# One predeclared nine-episode confirmation for the two-stage mechanism.
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-3000}"
WARMUP_STEPS="${WARMUP_STEPS:-3000}"
SEEDS="${SEEDS:-14100 14101 14102 14200 14201 14202 14300 14301 14302}"
ROOT="${ROOT:-data/derived/models/mature_weakness_probe}"

BASE_SEEDS="${BASE_SEEDS:-141 142 143}" EPISODES_PER_SEED=3 SEED_STRIDE=100 \
EPISODE_STEPS="$STEPS" WARMUP_STEPS="$WARMUP_STEPS" WIDTH="$WIDTH" HEIGHT="$HEIGHT" \
TARGET_FILL_LIMIT=470 TARGET_FILL_MODE=sources BOTTOM_LAYER_INTERVAL=20 \
DEPOSITION_PROBABILITY=0.45 DEPOSITION_MODE=sources SOURCE_COUNT=64 \
DAMAGE_ENABLED=1 DAMAGE_ACTIVATION_RATIO=0.85 DAMAGE_DECAY=0.985 DAMAGE_COUPLING=0.10 \
DAMAGE_THRESHOLD_REDUCTION=0.25 DAMAGE_RESET_FRACTION=0.90 \
MATURE_WEAKNESS_ENABLED=1 MATURE_WEAKNESS_DAMAGE_THRESHOLD=0.50 MATURE_WEAKNESS_DWELL_STEPS=5 \
MATURE_WEAKNESS_MATURATION_RATE=0.10 MATURE_WEAKNESS_DECAY=0.995 \
MATURE_WEAKNESS_THRESHOLD_REDUCTION=0.20 MATURE_WEAKNESS_RESET_FRACTION=0.90 \
EVENT_QUANTILE=0.998 EVENT_WINDOW=120 EVENT_MAX=5 RUN_HEATMAPS=0 RUN_EVENT_MAPS=0 \
RUN_SEED_MODELS=0 RUN_COMBINED=0 RUN_EVALUATIONS=0 ./scripts/run-synthetic-episode-batch.sh

mkdir -p "$ROOT"
for field in damage_total mature_weakness_total; do
  SEEDS="$SEEDS" WIDTH="$WIDTH" HEIGHT="$HEIGHT" STEPS="$STEPS" \
  ROOT="$ROOT/$field" PRIMARY_FIELD="$field" SIGNAL_FIELDS="$field" \
  LAG_EDGES="0 5 15 30 60" ./scripts/analyze-piezo-event-lead-time.sh
done

args=()
for seed in $SEEDS; do
  prefix="data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}"
  args+=(--piezo "${prefix}.piezo.csv" --events "${prefix}.avalanche_events.csv")
done
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-synthetic-step-targets \
  "${args[@]}" --out "$ROOT/targets.csv" --report "$ROOT/targets.json" --horizon-steps 15 --stride-steps 5
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli diagnose-synthetic-drift \
  --input "$ROOT/targets.csv" --out "$ROOT/drift.json" --target-field eventlist_target_occurred
