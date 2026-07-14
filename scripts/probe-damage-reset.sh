#!/usr/bin/env bash
set -euo pipefail

# Three-episode causal screen. Expand only profiles with supported lead time.
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-3000}"
WARMUP_STEPS="${WARMUP_STEPS:-3000}"
PROFILES="${PROFILES:-residual_damage rapid_reset}"
ROOT="${ROOT:-data/derived/models/damage_reset_probe}"

for profile in $PROFILES; do
  case "$profile" in
    residual_damage) base_seeds="132 133 134"; reset_fraction="0.75" ;;
    rapid_reset) base_seeds="135 136 137"; reset_fraction="0.98" ;;
    *) echo "error: unknown profile: $profile" >&2; exit 2 ;;
  esac
  seeds=""
  for base in $base_seeds; do seeds+=" $((base * 100))"; done
  seeds="${seeds# }"
  echo "damage reset profile: $profile (reset=$reset_fraction)"
  BASE_SEEDS="$base_seeds" EPISODES_PER_SEED=1 SEED_STRIDE=100 \
  EPISODE_STEPS="$STEPS" WARMUP_STEPS="$WARMUP_STEPS" WIDTH="$WIDTH" HEIGHT="$HEIGHT" \
  TARGET_FILL_LIMIT=470 TARGET_FILL_MODE=sources BOTTOM_LAYER_INTERVAL=20 \
  DEPOSITION_PROBABILITY=0.45 DEPOSITION_MODE=sources SOURCE_COUNT=64 \
  DAMAGE_ENABLED=1 DAMAGE_ACTIVATION_RATIO=0.85 DAMAGE_DECAY=0.985 DAMAGE_COUPLING=0.10 \
  DAMAGE_THRESHOLD_REDUCTION=0.25 DAMAGE_RESET_FRACTION="$reset_fraction" \
  EVENT_QUANTILE=0.998 EVENT_WINDOW=120 EVENT_MAX=5 RUN_HEATMAPS=0 RUN_EVENT_MAPS=0 \
  RUN_SEED_MODELS=0 RUN_COMBINED=0 RUN_EVALUATIONS=0 ./scripts/run-synthetic-episode-batch.sh
  SEEDS="$seeds" WIDTH="$WIDTH" HEIGHT="$HEIGHT" STEPS="$STEPS" \
  ROOT="$ROOT/$profile/lead_time" PRIMARY_FIELD=damage_total SIGNAL_FIELDS=damage_total \
  LAG_EDGES="0 5 15 30" ./scripts/analyze-piezo-event-lead-time.sh
done
