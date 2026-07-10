#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/synthetic_event_list_probes/h6/burn_0/targets.csv}"
ROOT="${ROOT:-data/derived/models/synthetic_event_list_patch_transformer}"
TRANSFORMER_INPUT="${TRANSFORMER_INPUT:-$ROOT/h6_transformer_input.csv}"
TRANSFORMER_INPUT_REPORT="${TRANSFORMER_INPUT_REPORT:-$ROOT/h6_transformer_input.json}"
OUT="${OUT:-$ROOT/h6_patch_transformer.json}"
SUMMARY="${SUMMARY:-$ROOT/h6_patch_transformer_summary.json}"
CHECKPOINT_OUT="${CHECKPOINT_OUT:-$ROOT/h6_patch_transformer.pt}"
SEEDS="${SEEDS:-10000 10001 10002 10100 10101 10102 10200 10201 10202}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-3000}"

LOOKBACK_STEPS="${LOOKBACK_STEPS:-12}"
PATCH_STEPS="${PATCH_STEPS:-3}"
D_MODEL="${D_MODEL:-64}"
LAYERS="${LAYERS:-3}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-24}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
RUN_SEEDS="${RUN_SEEDS:-42}"

if [[ ! -f "$INPUT" ]]; then
  echo "missing event-list target table: $INPUT" >&2
  echo "run ./scripts/probe-synthetic-event-list-models.sh first, or set INPUT" >&2
  exit 1
fi

mkdir -p "$ROOT"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli prepare-transformer-target-input \
  --input "$INPUT" \
  --out "$TRANSFORMER_INPUT" \
  --report "$TRANSFORMER_INPUT_REPORT" \
  --target-field eventlist_target_occurred \
  --target-status-field eventlist_target_status \
  --train-fraction "$TRAIN_FRACTION"

manifest_args=()
for seed in $SEEDS; do
  prefix="data/derived/models/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}"
  manifest_args+=(--sequence-manifest "${prefix}_avalanche_sequence/manifest.json")
  manifest_args+=(--sequence-manifest "${prefix}_piezo_sequence/manifest.json")
  manifest_args+=(--sequence-manifest "${prefix}_summary_sequence/manifest.json")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli train-torch-patch-transformer-split-holdout \
  --input "$TRANSFORMER_INPUT" \
  "${manifest_args[@]}" \
  --out "$OUT" \
  --lookback-steps "$LOOKBACK_STEPS" \
  --patch-steps "$PATCH_STEPS" \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --d-model "$D_MODEL" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --dropout "$DROPOUT" \
  --batch-size "$BATCH_SIZE" \
  --seed "$RUN_SEEDS" \
  --evaluation sequence_direct_avalanche_only \
  --evaluation sequence_piezo_vlf_only \
  --evaluation sequence_direct_avalanche_piezo_vlf \
  --evaluation sequence_full \
  --checkpoint-out "$CHECKPOINT_OUT"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-run-reports \
  --report "$OUT" \
  --out "$SUMMARY"

echo "summary: $SUMMARY"
