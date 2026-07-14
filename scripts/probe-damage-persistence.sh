#!/usr/bin/env bash
set -euo pipefail

# Compare only the damage-memory timescale. Other simulation and extraction
# settings match the existing delayed-failure target profile.
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-3000}"
WARMUP_STEPS="${WARMUP_STEPS:-3000}"
PROFILES="${PROFILES:-short_memory long_memory}"
ROOT="${ROOT:-data/derived/models/damage_persistence_probe}"

run_cli() {
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli "$@"
}

for profile in $PROFILES; do
  case "$profile" in
    short_memory)
      base_seeds="126 127 128"
      damage_decay="0.970"
      ;;
    long_memory)
      base_seeds="129 130 131"
      damage_decay="0.995"
      ;;
    *)
      echo "error: unknown profile: $profile" >&2
      exit 2
      ;;
  esac

  seeds=""
  for base_seed in $base_seeds; do
    for index in 0 1 2; do
      seeds+=" $((base_seed * 100 + index))"
    done
  done
  seeds="${seeds# }"
  profile_root="$ROOT/$profile"
  mkdir -p "$profile_root"

  echo "damage persistence profile: $profile (decay=$damage_decay)"
  BASE_SEEDS="$base_seeds" \
  EPISODES_PER_SEED=3 \
  SEED_STRIDE=100 \
  EPISODE_STEPS="$STEPS" \
  WARMUP_STEPS="$WARMUP_STEPS" \
  WIDTH="$WIDTH" HEIGHT="$HEIGHT" \
  TARGET_FILL_LIMIT=470 TARGET_FILL_MODE=sources \
  BOTTOM_LAYER_INTERVAL=20 DEPOSITION_PROBABILITY=0.45 DEPOSITION_MODE=sources SOURCE_COUNT=64 \
  DAMAGE_ENABLED=1 DAMAGE_ACTIVATION_RATIO=0.85 DAMAGE_DECAY="$damage_decay" \
  DAMAGE_COUPLING=0.10 DAMAGE_THRESHOLD_REDUCTION=0.25 DAMAGE_RESET_FRACTION=0.90 \
  EVENT_QUANTILE=0.998 EVENT_WINDOW=120 EVENT_MAX=5 \
  RUN_HEATMAPS=0 RUN_EVENT_MAPS=0 RUN_SEED_MODELS=1 RUN_COMBINED=0 RUN_EVALUATIONS=0 \
    ./scripts/run-synthetic-episode-batch.sh

  SEEDS="$seeds" WIDTH="$WIDTH" HEIGHT="$HEIGHT" STEPS="$STEPS" \
  ROOT="$profile_root/lead_time" PRIMARY_FIELD=damage_total SIGNAL_FIELDS=damage_total \
  LAG_EDGES="0 5 15 30" ./scripts/analyze-piezo-event-lead-time.sh

  step_args=()
  manifest_args=()
  for seed in $seeds; do
    prefix="data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}"
    model_prefix="data/derived/models/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}"
    step_args+=(--piezo "${prefix}.piezo.csv" --events "${prefix}.avalanche_events.csv")
    manifest_args+=(--piezo-sequence-manifest "${model_prefix}_piezo_sequence/manifest.json")
  done
  run_cli build-synthetic-step-targets "${step_args[@]}" \
    --out "$profile_root/targets.csv" --report "$profile_root/targets.json" \
    --horizon-steps 15 --stride-steps 5
  run_cli diagnose-synthetic-drift --input "$profile_root/targets.csv" \
    --out "$profile_root/drift.json" --target-field eventlist_target_occurred
  run_cli evaluate-damage-precursor-head --target "$profile_root/targets.csv" \
    "${manifest_args[@]}" --out "$profile_root/damage_head_60m.json" \
    --short-steps 5 --long-steps 60 --epochs 20 --learning-rate 0.05 --l2 0.01
done
