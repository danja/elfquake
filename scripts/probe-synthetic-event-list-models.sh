#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BASE="${BASE:-data/derived/models/mountain_256x256_seeds10000-10202_3000}"
ALIGNED="${ALIGNED:-${BASE}.aligned_hourly_synthetic_windows.csv}"
ROOT="${ROOT:-data/derived/models/synthetic_event_list_probes}"
HORIZONS="${HORIZONS:-3 6 12}"
BURN_IN_FRACTIONS="${BURN_IN_FRACTIONS:-0 0.2}"
REGIME_COUNT="${REGIME_COUNT:-5}"
RUN_BALANCED_CONTROLS="${RUN_BALANCED_CONTROLS:-1}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
EPOCHS="${EPOCHS:-600}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
L2="${L2:-0.001}"

if [[ ! -f "$ALIGNED" ]]; then
  echo "missing aligned synthetic windows: $ALIGNED" >&2
  exit 1
fi

mkdir -p "$ROOT"

slug_fraction() {
  local value="$1"
  value="${value//./p}"
  echo "$value"
}

is_zero_fraction() {
  local value="$1"
  [[ "$value" == "0" || "$value" == "0.0" || "$value" == "0.00" ]]
}

run_model() {
  local input="$1"
  local split_field="$2"
  local output_dir="$3"
  local name="$4"
  local model_type="$5"
  local ensemble_count="$6"
  local bag_fraction="$7"
  local max_features="$8"
  local stump_count="$9"

  mkdir -p "$output_dir"
  INPUT="$input" \
  OUT="${output_dir}/${name}.json" \
  PREDICTIONS_OUT="${output_dir}/${name}.predictions.csv" \
  SPLIT_FIELD="$split_field" \
  TRAIN_FRACTION="$TRAIN_FRACTION" \
  EPOCHS="$EPOCHS" \
  LEARNING_RATE="$LEARNING_RATE" \
  L2="$L2" \
  OCCURRENCE_MODEL_TYPE="$model_type" \
  OCCURRENCE_ENSEMBLE_COUNT="$ensemble_count" \
  OCCURRENCE_FEATURE_BAG_FRACTION="$bag_fraction" \
  OCCURRENCE_STUMP_COUNT="$stump_count" \
  MAX_FEATURE_COUNT="$max_features" \
    ./scripts/train-synthetic-event-list-model.sh
}

for horizon in $HORIZONS; do
  for burn_in_fraction in $BURN_IN_FRACTIONS; do
    burn_slug="$(slug_fraction "$burn_in_fraction")"
    run_dir="${ROOT}/h${horizon}/burn_${burn_slug}"
    mkdir -p "$run_dir"

    targets="${run_dir}/targets.csv"
    target_report="${run_dir}/targets.json"
    if is_zero_fraction "$burn_in_fraction"; then
      INPUT="$ALIGNED" \
      OUT="$targets" \
      REPORT="$target_report" \
      HORIZON_ROWS="$horizon" \
        ./scripts/build-synthetic-event-list-targets.sh
    else
      raw_targets="${run_dir}/targets.pre_burn_in.csv"
      raw_report="${run_dir}/targets.pre_burn_in.json"
      INPUT="$ALIGNED" \
      OUT="$raw_targets" \
      REPORT="$raw_report" \
      HORIZON_ROWS="$horizon" \
        ./scripts/build-synthetic-event-list-targets.sh

      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli annotate-synthetic-regimes \
        --input "$raw_targets" \
        --out "$targets" \
        --report "$target_report" \
        --regime-count "$REGIME_COUNT" \
        --burn-in-fraction "$burn_in_fraction" \
        --drop-burn-in
    fi

    INPUT="$targets" \
    OUT="${run_dir}/drift.json" \
    CSV_OUT="${run_dir}/drift_buckets.csv" \
      ./scripts/diagnose-synthetic-event-list-drift.sh

    episodes="${run_dir}/targets.episodes.csv"
    INPUT="$targets" \
    OUT="$episodes" \
    REPORT="${run_dir}/targets.episodes.json" \
      ./scripts/annotate-synthetic-event-list-episodes.sh

    balanced="${run_dir}/targets.balanced_split.csv"
    INPUT="$targets" \
    OUT="$balanced" \
    REPORT="${run_dir}/targets.balanced_split.json" \
    GROUP_FIELD=dataset_id \
    TARGET_FIELD=eventlist_target_occurred \
      ./scripts/build-synthetic-event-list-split.sh

    episode_balanced="${run_dir}/targets.episodes.balanced_split.csv"
    INPUT="$episodes" \
    OUT="$episode_balanced" \
    REPORT="${run_dir}/targets.episodes.balanced_split.json" \
    GROUP_FIELD=synthetic_episode_id \
    TARGET_FIELD=eventlist_target_occurred \
      ./scripts/build-synthetic-event-list-split.sh

    models_dir="${run_dir}/models"
    run_model "$targets" "" "$models_dir" "temporal_ensemble_default" "logistic_ensemble" "8" "0.5" "0" "24"
    run_model "$targets" "" "$models_dir" "temporal_ensemble_16_bag_0p4" "logistic_ensemble" "16" "0.4" "0" "24"
    run_model "$targets" "" "$models_dir" "temporal_feature_cap_64" "logistic_ensemble" "8" "0.5" "64" "24"
    run_model "$targets" "" "$models_dir" "temporal_feature_cap_32" "logistic_ensemble" "8" "0.5" "32" "24"
    run_model "$targets" "" "$models_dir" "temporal_boosted_stumps_24" "boosted_stumps" "8" "0.5" "0" "24"

    if [[ "$RUN_BALANCED_CONTROLS" != "0" ]]; then
      run_model "$balanced" "model_split" "$models_dir" "balanced_ensemble_default" "logistic_ensemble" "8" "0.5" "0" "24"
      run_model "$episode_balanced" "model_split" "$models_dir" "episode_balanced_ensemble_default" "logistic_ensemble" "8" "0.5" "0" "24"
      run_model "$balanced" "model_split" "$models_dir" "balanced_boosted_stumps_24" "boosted_stumps" "8" "0.5" "0" "24"
    fi
  done
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-synthetic-event-list-probes \
  --root "$ROOT" \
  --out "${ROOT}/summary.json" \
  --csv-out "${ROOT}/summary.csv"

echo "summary: ${ROOT}/summary.csv"
