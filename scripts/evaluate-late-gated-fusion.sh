#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/synthetic_event_list_probes/h6/burn_0/targets.csv}"
ROOT="${ROOT:-data/derived/models/late_gated_fusion}"
TRANSFORMER_INPUT="${TRANSFORMER_INPUT:-$ROOT/h6_transformer_input.csv}"
TRANSFORMER_INPUT_REPORT="${TRANSFORMER_INPUT_REPORT:-$ROOT/h6_transformer_input.json}"
OUT="${OUT:-$ROOT/evaluation.json}"
SEEDS="${SEEDS:-10000 10001 10002 10100 10101 10102 10200 10201 10202}"
RUN_SEEDS="${RUN_SEEDS:-7 42 99}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-3000}"

mkdir -p "$ROOT"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli prepare-transformer-target-input \
  --input "$INPUT" \
  --out "$TRANSFORMER_INPUT" \
  --report "$TRANSFORMER_INPUT_REPORT" \
  --target-field eventlist_target_occurred \
  --target-status-field eventlist_target_status \
  --train-fraction "${TRAIN_FRACTION:-0.8}"

manifest_args=()
for seed in $SEEDS; do
  prefix="data/derived/models/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}"
  manifest_args+=(--synthetic-sequence-manifest "${prefix}_avalanche_sequence/manifest.json")
  manifest_args+=(--synthetic-sequence-manifest "${prefix}_piezo_sequence/manifest.json")
  manifest_args+=(--synthetic-sequence-manifest "${prefix}_summary_sequence/manifest.json")
done

seed_args=()
for seed in $RUN_SEEDS; do
  seed_args+=(--seed "$seed")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-late-gated-fusion \
  --target "$TRANSFORMER_INPUT" \
  "${manifest_args[@]}" \
  --out "$OUT" \
  --artifact-root "$ROOT/checkpoints" \
  "${seed_args[@]}" \
  --lookback-steps "${LOOKBACK_STEPS:-12}" \
  --patch-steps "${PATCH_STEPS:-3}" \
  --pretrain-stride "${PRETRAIN_STRIDE:-3}" \
  --ssl-epochs "${SSL_EPOCHS:-6}" \
  --supervised-epochs "${SUPERVISED_EPOCHS:-12}" \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --d-model "${D_MODEL:-32}" \
  --layers "${LAYERS:-2}" \
  --heads "${HEADS:-4}" \
  --dropout "${DROPOUT:-0.1}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --mask-probability "${MASK_PROBABILITY:-0.30}" \
  --modality-dropout-probability "${MODALITY_DROPOUT_PROBABILITY:-0.25}" \
  --max-pretrain-windows "${MAX_PRETRAIN_WINDOWS:-2048}"
