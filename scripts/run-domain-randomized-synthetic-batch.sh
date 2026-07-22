#!/usr/bin/env bash
set -euo pipefail

# Keep profiles explicit and disjoint in seed space so each episode remains
# traceable to one simulator regime.
EPISODE_STEPS="${EPISODE_STEPS:-3000}"
EPISODES_PER_PROFILE="${EPISODES_PER_PROFILE:-2}"
ROOT="${ROOT:-data/derived/models/domain_randomized_batch}"
mkdir -p "$ROOT"

profiles=(
  "baseline|140 141|0.45|25|1024|0"
  "slow_fill|150 151|0.30|40|512|0"
  "fast_localized|160 161|0.65|20|2048|0"
)

for profile in "${profiles[@]}"; do
  IFS='|' read -r name seeds deposition removal sources damage <<< "$profile"
  printf '%s\n' "profile=$name" "base_seeds=$seeds" "episode_steps=$EPISODE_STEPS" \
    "deposition_probability=$deposition" "bottom_layer_interval=$removal" \
    "source_count=$sources" "damage_enabled=$damage" > "$ROOT/$name.profile.txt"
  BASE_SEEDS="$seeds" \
  EPISODES_PER_SEED="$EPISODES_PER_PROFILE" \
  EPISODE_STEPS="$EPISODE_STEPS" \
  DEPOSITION_PROBABILITY="$deposition" \
  BOTTOM_LAYER_INTERVAL="$removal" \
  SOURCE_COUNT="$sources" \
  DAMAGE_ENABLED="$damage" \
  RUN_HEATMAPS=0 \
  RUN_EVALUATIONS=0 \
    ./scripts/run-synthetic-episode-batch.sh
done

echo "domain-randomized batch: $ROOT"
