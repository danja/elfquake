#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./refresh-synthetic-model-artifacts.sh

Refreshes derived synthetic event, map, aligned model, tensor, and smoke-report
artifacts from existing simulation CSVs.

Environment:
  SEEDS             default "40 41 42"
  WIDTH             default 256
  HEIGHT            default 256
  STEPS             default 10000
  REGION_ID         default central_italy
  START_UTC         default 2026-01-01T00:00:00Z
  END_UTC           default 2026-01-08T00:00:00Z
  WINDOW_SECONDS    default 3600
  EVENT_QUANTILE    default 0.99
  EVENT_WINDOW      default 30
  EVENT_MAX         default 0, no cap
  SPATIAL_PROFILE   default italy_apennines
  OUTPUT_TAG        optional suffix for parallel artifacts, e.g. _sparse

  RUN_EVENT_LISTS   default 1
  RUN_EVENT_MAPS    default 1
  RUN_SEED_MODELS   default 1
  RUN_COMBINED      default 1
  RUN_EVALUATIONS   default 1
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

seeds="${SEEDS:-40 41 42}"
width="${WIDTH:-256}"
height="${HEIGHT:-256}"
steps="${STEPS:-10000}"
region_id="${REGION_ID:-central_italy}"
start_utc="${START_UTC:-2026-01-01T00:00:00Z}"
end_utc="${END_UTC:-2026-01-08T00:00:00Z}"
window_seconds="${WINDOW_SECONDS:-3600}"
event_quantile="${EVENT_QUANTILE:-0.99}"
event_window="${EVENT_WINDOW:-30}"
event_max="${EVENT_MAX:-0}"
spatial_profile="${SPATIAL_PROFILE:-italy_apennines}"
output_tag="${OUTPUT_TAG:-}"
run_event_lists="${RUN_EVENT_LISTS:-1}"
run_event_maps="${RUN_EVENT_MAPS:-1}"
run_seed_models="${RUN_SEED_MODELS:-1}"
run_combined="${RUN_COMBINED:-1}"
run_evaluations="${RUN_EVALUATIONS:-1}"

read -r -a seed_list <<< "$seeds"
last_seed="${seed_list[$((${#seed_list[@]} - 1))]}"
if [[ "${#seed_list[@]}" -eq 1 ]]; then
  seed_label="${seed_list[0]}"
else
  seed_label="${seed_list[0]}-${last_seed}"
fi

run_cli() {
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "error: required file not found: $1" >&2
    exit 2
  fi
}

for seed in $seeds; do
  prefix="mountain_${width}x${height}_seed${seed}_${steps}"
  sim_prefix="data/derived/sim/${prefix}"
  event_file="${sim_prefix}.avalanche_events${output_tag}.csv"
  model_prefix="data/derived/models/${prefix}${output_tag}"
  echo "refresh seed: $seed"
  require_file "${sim_prefix}.summary.csv"
  require_file "${sim_prefix}.sensors.csv"
  require_file "${sim_prefix}.piezo.csv"
  require_file "${sim_prefix}.avalanche_signal.csv"
  require_file "${sim_prefix}.avalanche_activity.csv"

  if [[ "$run_event_lists" != "0" ]]; then
    run_cli build-synthetic-event-list \
      --summary "${sim_prefix}.summary.csv" \
      --sensors "${sim_prefix}.sensors.csv" \
      --grid-width "$width" \
      --grid-height "$height" \
      --out "${sim_prefix}.synthetic_events.csv"

    run_cli build-avalanche-signal-event-list \
      --avalanche "${sim_prefix}.avalanche_signal.csv" \
      --activity "${sim_prefix}.avalanche_activity.csv" \
      --grid-width "$width" \
      --grid-height "$height" \
      --min-signal-quantile "$event_quantile" \
      --local-max-window "$event_window" \
      --max-events "$event_max" \
      --spatial-profile "$spatial_profile" \
      --out "$event_file"
  fi

  if [[ "$run_event_maps" != "0" ]]; then
    ./event-map.sh "$event_file"
  fi

  if [[ "$run_seed_models" != "0" ]]; then
    require_file "$event_file"
    run_cli build-event-window-features \
      --events "$event_file" \
      --out "${model_prefix}.hourly_synthetic_seismic_windows.csv" \
      --region-id "$region_id" \
      --start-utc "$start_utc" \
      --end-utc "$end_utc" \
      --window-seconds "$window_seconds" \
      --feature-prefix synthetic_seismic

    run_cli build-tensor-spec \
      --input "${model_prefix}.hourly_synthetic_seismic_windows.csv" \
      --out "${model_prefix}.hourly_synthetic_seismic_windows_tensor_spec.json" \
      --time-field window_start_utc \
      --region-field region_id \
      --target-field target_occurred

    run_cli materialize-tensor-dataset \
      --spec "${model_prefix}.hourly_synthetic_seismic_windows_tensor_spec.json" \
      --out-dir "${model_prefix}_hourly_synthetic_seismic_windows_tensor"

    run_cli materialize-sequence-dataset \
      --input "${sim_prefix}.piezo.csv" \
      --out-dir "${model_prefix}_piezo_sequence" \
      --time-field step \
      --entity-field sensor_id \
      --modality synthetic_piezo_vlf \
      --time-start-utc "$start_utc" \
      --time-step-seconds 60

    run_cli materialize-sequence-dataset \
      --input "${sim_prefix}.avalanche_signal.csv" \
      --out-dir "${model_prefix}_avalanche_sequence" \
      --time-field step \
      --entity-field sensor_id \
      --modality synthetic_direct_avalanche \
      --time-start-utc "$start_utc" \
      --time-step-seconds 60

    run_cli materialize-sequence-dataset \
      --input "${sim_prefix}.summary.csv" \
      --out-dir "${model_prefix}_summary_sequence" \
      --time-field step \
      --no-entity-field \
      --modality synthetic_summary \
      --time-start-utc "$start_utc" \
      --time-step-seconds 60

    for suffix in "" "_gt1"; do
      threshold="0"
      if [[ "$suffix" == "_gt1" ]]; then
        threshold="1"
      fi
      run_cli build-aligned-window-dataset \
        --base-manifest "${model_prefix}_hourly_synthetic_seismic_windows_tensor/manifest.json" \
        --sequence-manifest "${model_prefix}_piezo_sequence/manifest.json" \
        --sequence-manifest "${model_prefix}_avalanche_sequence/manifest.json" \
        --sequence-manifest "${model_prefix}_summary_sequence/manifest.json" \
        --target-source-feature synthetic_seismic_event_count \
        --target-horizon-rows 1 \
        --target-threshold "$threshold" \
        --drop-unlabeled-targets \
        --out "${model_prefix}.aligned_hourly_synthetic_windows${suffix}.csv"

      run_cli build-tensor-spec \
        --input "${model_prefix}.aligned_hourly_synthetic_windows${suffix}.csv" \
        --out "${model_prefix}.aligned_hourly_synthetic_windows${suffix}_tensor_spec.json" \
        --time-field window_start_utc \
        --region-field region_id \
        --target-field target_occurred

      run_cli materialize-tensor-dataset \
        --spec "${model_prefix}.aligned_hourly_synthetic_windows${suffix}_tensor_spec.json" \
        --out-dir "${model_prefix}_aligned_hourly_synthetic_windows${suffix}_tensor"
    done
  fi
done

if [[ "$run_combined" != "0" ]]; then
  combined_prefix="data/derived/models/mountain_${width}x${height}_seeds${seed_label}_${steps}${output_tag}"
  inputs=()
  gt1_inputs=()
  for seed in $seeds; do
    prefix="data/derived/models/mountain_${width}x${height}_seed${seed}_${steps}${output_tag}"
    inputs+=(--input "${prefix}.aligned_hourly_synthetic_windows.csv" --dataset-id "seed${seed}")
    gt1_inputs+=(--input "${prefix}.aligned_hourly_synthetic_windows_gt1.csv" --dataset-id "seed${seed}")
  done

  run_cli combine-aligned-datasets "${inputs[@]}" \
    --out "${combined_prefix}.aligned_hourly_synthetic_windows.csv"
  run_cli combine-aligned-datasets "${gt1_inputs[@]}" \
    --out "${combined_prefix}.aligned_hourly_synthetic_windows_gt1.csv"

  for suffix in "" "_gt1"; do
    run_cli build-tensor-spec \
      --input "${combined_prefix}.aligned_hourly_synthetic_windows${suffix}.csv" \
      --out "${combined_prefix}.aligned_hourly_synthetic_windows${suffix}_tensor_spec.json" \
      --time-field window_start_utc \
      --region-field region_id \
      --target-field target_occurred
    run_cli materialize-tensor-dataset \
      --spec "${combined_prefix}.aligned_hourly_synthetic_windows${suffix}_tensor_spec.json" \
      --out-dir "${combined_prefix}_aligned_hourly_synthetic_windows${suffix}_tensor"
  done
fi

if [[ "$run_evaluations" != "0" ]]; then
  combined_prefix="data/derived/models/mountain_${width}x${height}_seeds${seed_label}_${steps}${output_tag}"
  for suffix in "" "_gt1"; do
    base="${combined_prefix}.aligned_hourly_synthetic_windows${suffix}"
    run_cli summarize-model-readiness --input "${base}.csv" --out "${base}.readiness.json"
    run_cli train-ablation-smoke --input "${base}.csv" --out "${base}.ablation_smoke.json"
    run_cli evaluate-temporal-holdout --input "${base}.csv" --out "${base}.temporal_holdout.json" --train-fraction 0.8

    reports=(--report "${base}.ablation_smoke.json" --report "${base}.temporal_holdout.json")
    for seed in $seeds; do
      run_cli evaluate-group-holdout \
        --input "${base}.csv" \
        --out "${base}.group_holdout_seed${seed}.json" \
        --group-field dataset_id \
        --test-group "seed${seed}"
      reports+=(--report "${base}.group_holdout_seed${seed}.json")
    done
    run_cli summarize-model-run-reports "${reports[@]}" --out "${base}.model_run_summary.json"
  done
fi

echo "done"
