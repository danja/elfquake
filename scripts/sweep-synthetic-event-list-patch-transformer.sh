#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/synthetic_event_list_probes/h6/burn_0/targets.csv}"
ROOT="${ROOT:-data/derived/models/synthetic_event_list_patch_transformer_sweep}"
LOOKBACK_STEPS_LIST="${LOOKBACK_STEPS_LIST:-8 12}"
PATCH_STEPS_LIST="${PATCH_STEPS_LIST:-2 3}"
DROPOUTS="${DROPOUTS:-0.1}"
EPOCHS="${EPOCHS:-12}"
RUN_SEEDS="${RUN_SEEDS:-42}"

mkdir -p "$ROOT"

slug_number() {
  local value="$1"
  value="${value//./p}"
  echo "$value"
}

reports=()
for lookback_steps in $LOOKBACK_STEPS_LIST; do
  for patch_steps in $PATCH_STEPS_LIST; do
    if (( patch_steps > lookback_steps )); then
      continue
    fi
    for dropout in $DROPOUTS; do
      dropout_slug="$(slug_number "$dropout")"
      config_dir="${ROOT}/lookback_${lookback_steps}/patch_${patch_steps}/dropout_${dropout_slug}"
      for run_seed in $RUN_SEEDS; do
        run_dir="${config_dir}/seed_${run_seed}"
        mkdir -p "$run_dir"
        INPUT="$INPUT" \
        ROOT="$run_dir" \
        OUT="${run_dir}/patch_transformer.json" \
        SUMMARY="${run_dir}/patch_transformer_summary.json" \
        CHECKPOINT_OUT="${run_dir}/patch_transformer.pt" \
        LOOKBACK_STEPS="$lookback_steps" \
        PATCH_STEPS="$patch_steps" \
        DROPOUT="$dropout" \
        EPOCHS="$EPOCHS" \
        RUN_SEEDS="$run_seed" \
          ./scripts/train-synthetic-event-list-patch-transformer.sh
        reports+=(--report "${run_dir}/patch_transformer.json")
      done
    done
  done
done

if (( ${#reports[@]} > 0 )); then
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-run-reports \
    "${reports[@]}" \
    --out "${ROOT}/summary.json"
  echo "summary: ${ROOT}/summary.json"
fi
