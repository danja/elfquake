#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-data/derived/models/mountain_256x256_seeds40-42_20000}"
ALIGNED="${ALIGNED:-${BASE}.aligned_hourly_synthetic_windows.csv}"
TARGETS="${TARGETS:-${BASE}.synthetic_event_list_targets_h6.csv}"
TARGET_REPORT="${TARGET_REPORT:-${BASE}.synthetic_event_list_targets_h6.json}"
PRE_BURN_IN_TARGETS="${PRE_BURN_IN_TARGETS:-${BASE}.synthetic_event_list_targets_h6.pre_burn_in.csv}"
PRE_BURN_IN_REPORT="${PRE_BURN_IN_REPORT:-${BASE}.synthetic_event_list_targets_h6.pre_burn_in.json}"
BURN_IN_REPORT="${BURN_IN_REPORT:-${BASE}.synthetic_event_list_targets_h6.burn_in.json}"
BURN_IN_FRACTION="${BURN_IN_FRACTION:-0}"
REGIME_COUNT="${REGIME_COUNT:-5}"
GROUP_FIELD="${GROUP_FIELD:-dataset_id}"
TIME_FIELD="${TIME_FIELD:-window_start_utc}"
EPISODES="${EPISODES:-${BASE}.synthetic_event_list_targets_h6.episodes.csv}"
BALANCED="${BALANCED:-${BASE}.synthetic_event_list_targets_h6.balanced_split.csv}"
EPISODE_BALANCED="${EPISODE_BALANCED:-${BASE}.synthetic_event_list_targets_h6.episodes.balanced_split.csv}"
DRIFT_DIR="${DRIFT_DIR:-data/derived/models/synthetic_event_list_drift}"
MODEL_DIR="${MODEL_DIR:-data/derived/models/synthetic_event_list_model}"

if [[ "$BURN_IN_FRACTION" == "0" || "$BURN_IN_FRACTION" == "0.0" ]]; then
  INPUT="$ALIGNED" \
  OUT="$TARGETS" \
  REPORT="$TARGET_REPORT" \
    ./scripts/build-synthetic-event-list-targets.sh
else
  INPUT="$ALIGNED" \
  OUT="$PRE_BURN_IN_TARGETS" \
  REPORT="$PRE_BURN_IN_REPORT" \
    ./scripts/build-synthetic-event-list-targets.sh

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli annotate-synthetic-regimes \
    --input "$PRE_BURN_IN_TARGETS" \
    --out "$TARGETS" \
    --report "$BURN_IN_REPORT" \
    --group-field "$GROUP_FIELD" \
    --time-field "$TIME_FIELD" \
    --regime-count "$REGIME_COUNT" \
    --burn-in-fraction "$BURN_IN_FRACTION" \
    --drop-burn-in
fi

INPUT="$TARGETS" \
OUT="${DRIFT_DIR}/h6_drift.json" \
CSV_OUT="${DRIFT_DIR}/h6_drift_buckets.csv" \
  ./scripts/diagnose-synthetic-event-list-drift.sh

INPUT="$TARGETS" \
OUT="$EPISODES" \
REPORT="${BASE}.synthetic_event_list_targets_h6.episodes.json" \
  ./scripts/annotate-synthetic-event-list-episodes.sh

INPUT="$TARGETS" \
OUT="$BALANCED" \
REPORT="${BASE}.synthetic_event_list_targets_h6.balanced_split.json" \
GROUP_FIELD="$GROUP_FIELD" \
TARGET_FIELD=eventlist_target_occurred \
  ./scripts/build-synthetic-event-list-split.sh

INPUT="$EPISODES" \
OUT="$EPISODE_BALANCED" \
REPORT="${BASE}.synthetic_event_list_targets_h6.episodes.balanced_split.json" \
GROUP_FIELD=synthetic_episode_id \
TARGET_FIELD=eventlist_target_occurred \
  ./scripts/build-synthetic-event-list-split.sh

INPUT="$TARGETS" \
OUT="${MODEL_DIR}/h6_event_list_model.json" \
PREDICTIONS_OUT="${MODEL_DIR}/h6_event_list_predictions.csv" \
  ./scripts/train-synthetic-event-list-model.sh

INPUT="$BALANCED" \
OUT="${MODEL_DIR}/h6_balanced_event_list_model.json" \
PREDICTIONS_OUT="${MODEL_DIR}/h6_balanced_event_list_predictions.csv" \
SPLIT_FIELD=model_split \
  ./scripts/train-synthetic-event-list-model.sh

INPUT="$EPISODE_BALANCED" \
OUT="${MODEL_DIR}/h6_episode_balanced_event_list_model.json" \
PREDICTIONS_OUT="${MODEL_DIR}/h6_episode_balanced_event_list_predictions.csv" \
SPLIT_FIELD=model_split \
  ./scripts/train-synthetic-event-list-model.sh
