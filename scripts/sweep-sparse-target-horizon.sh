#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/sweep-sparse-target-horizon.sh

Builds sparse-profile aligned synthetic rows for several target horizons and
runs readiness plus temporal/group smoke reports.

Environment:
  SEEDS          default "40 41 42"
  WIDTH          default 256
  HEIGHT         default 256
  STEPS          default 20000
  HORIZONS       default "1 3 6 12 24"
  OUTPUT_TAG     default _sparse
  WINDOW_SECONDS default 3600
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

seeds="${SEEDS:-40 41 42}"
width="${WIDTH:-256}"
height="${HEIGHT:-256}"
steps="${STEPS:-20000}"
horizons="${HORIZONS:-1 3 6 12 24}"
output_tag="${OUTPUT_TAG:-_sparse}"

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

for horizon in $horizons; do
  echo "horizon rows: $horizon"
  inputs=()
  for seed in $seeds; do
    prefix="data/derived/models/mountain_${width}x${height}_seed${seed}_${steps}${output_tag}"
    out="${prefix}.aligned_hourly_synthetic_windows_h${horizon}.csv"
    require_file "${prefix}_hourly_synthetic_seismic_windows_tensor/manifest.json"
    require_file "${prefix}_piezo_sequence/manifest.json"
    require_file "${prefix}_avalanche_sequence/manifest.json"
    require_file "${prefix}_summary_sequence/manifest.json"
    run_cli build-aligned-window-dataset \
      --base-manifest "${prefix}_hourly_synthetic_seismic_windows_tensor/manifest.json" \
      --sequence-manifest "${prefix}_piezo_sequence/manifest.json" \
      --sequence-manifest "${prefix}_avalanche_sequence/manifest.json" \
      --sequence-manifest "${prefix}_summary_sequence/manifest.json" \
      --target-source-feature synthetic_seismic_event_count \
      --target-horizon-rows "$horizon" \
      --target-threshold 0 \
      --drop-unlabeled-targets \
      --out "$out"
    inputs+=(--input "$out" --dataset-id "seed${seed}")
  done

  combined="data/derived/models/mountain_${width}x${height}_seeds${seed_label}_${steps}${output_tag}.aligned_hourly_synthetic_windows_h${horizon}"
  run_cli combine-aligned-datasets "${inputs[@]}" --out "${combined}.csv"
  run_cli summarize-model-readiness --input "${combined}.csv" --out "${combined}.readiness.json"
  run_cli evaluate-temporal-holdout --input "${combined}.csv" --out "${combined}.temporal_holdout.json" --train-fraction 0.8

  reports=(--report "${combined}.temporal_holdout.json")
  for seed in $seeds; do
    run_cli evaluate-group-holdout \
      --input "${combined}.csv" \
      --out "${combined}.group_holdout_seed${seed}.json" \
      --group-field dataset_id \
      --test-group "seed${seed}"
    reports+=(--report "${combined}.group_holdout_seed${seed}.json")
  done
  run_cli summarize-model-run-reports "${reports[@]}" --out "${combined}.model_run_summary.json"
done

echo "done"
