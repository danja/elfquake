#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/synthetic_event_list_probes/h6/burn_0/targets.csv}"
ROOT="${ROOT:-data/derived/models/synthetic_event_list_patch_transformer_episode_holdout}"
SEEDS="${SEEDS:-10000 10001 10002 10100 10101 10102 10200 10201 10202}"
LOOKBACK_STEPS="${LOOKBACK_STEPS:-8}"
PATCH_STEPS="${PATCH_STEPS:-3}"
EPOCHS="${EPOCHS:-12}"
MODEL_SEED="${MODEL_SEED:-42}"
SEQUENCE_NORMALIZATION="${SEQUENCE_NORMALIZATION:-global}"
REGRESSION_TARGETS="${REGRESSION_TARGETS-eventlist_target_count eventlist_target_log10_magnitude_energy}"

mkdir -p "$ROOT"
mapfile -t groups < <(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" - "$INPUT" <<'PY'
import csv, sys
seen = set()
with open(sys.argv[1], newline='', encoding='utf-8') as handle:
    for row in csv.DictReader(handle):
        group = row.get('dataset_id', '')
        if group and group not in seen:
            seen.add(group)
            print(group)
PY
)

reports=()
for group in "${groups[@]}"; do
  slug="${group//[^A-Za-z0-9_.-]/_}"
  fold_root="$ROOT/$slug"
  input_out="$fold_root/transformer_input.csv"
  mkdir -p "$fold_root"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli prepare-transformer-target-input \
    --input "$INPUT" \
    --out "$input_out" \
    --report "$fold_root/transformer_input.json" \
    --target-field eventlist_target_occurred \
    --target-status-field eventlist_target_status \
    --test-group "$group"

  manifest_args=()
  for seed in $SEEDS; do
    prefix="data/derived/models/mountain_256x256_seed${seed}_3000"
    manifest_args+=(--sequence-manifest "${prefix}_avalanche_sequence/manifest.json")
    manifest_args+=(--sequence-manifest "${prefix}_piezo_sequence/manifest.json")
    manifest_args+=(--sequence-manifest "${prefix}_summary_sequence/manifest.json")
  done
  regression_args=()
  for target in $REGRESSION_TARGETS; do
    regression_args+=(--regression-target "$target")
  done
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli train-torch-patch-transformer-split-holdout \
    --input "$input_out" "${manifest_args[@]}" \
    --out "$fold_root/patch_transformer.json" \
    --lookback-steps "$LOOKBACK_STEPS" \
    --patch-steps "$PATCH_STEPS" \
    --epochs "$EPOCHS" \
    --seed "$MODEL_SEED" \
    --sequence-normalization "$SEQUENCE_NORMALIZATION" \
    --evaluation sequence_piezo_vlf_only \
    "${regression_args[@]}"
  reports+=(--report "$fold_root/patch_transformer.json")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-run-reports \
  "${reports[@]}" --out "$ROOT/summary.json"
echo "summary: $ROOT/summary.json"
