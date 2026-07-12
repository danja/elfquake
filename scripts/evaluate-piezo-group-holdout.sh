#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/synthetic_event_list_probes/h6/burn_0/targets.csv}"
ROOT="${ROOT:-data/derived/models/piezo_group_holdout}"
TRANSFORMER_INPUT="${TRANSFORMER_INPUT:-$ROOT/transformer_input.csv}"
TRANSFORMER_INPUT_REPORT="${TRANSFORMER_INPUT_REPORT:-$ROOT/transformer_input.json}"
OUT="${OUT:-$ROOT/evaluation.json}"
SEEDS="${SEEDS:-10000 10001 10002 10100 10101 10102 10200 10201 10202}"
RUN_SEEDS="${RUN_SEEDS:-7 42 99}"
EXCLUDE_PIEZO_FIELDS="${EXCLUDE_PIEZO_FIELDS:-}"
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
  manifest_args+=(--piezo-sequence-manifest "${prefix}_piezo_sequence/manifest.json")
done

seed_args=()
for seed in $RUN_SEEDS; do
  seed_args+=(--seed "$seed")
done

exclude_args=()
for field in $EXCLUDE_PIEZO_FIELDS; do
  exclude_args+=(--exclude-piezo-field "$field")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-piezo-group-holdout \
  --target "$TRANSFORMER_INPUT" \
  "${manifest_args[@]}" \
  --out "$OUT" \
  --artifact-root "$ROOT/checkpoints" \
  "${seed_args[@]}" \
  "${exclude_args[@]}" \
  --lookback-steps "${LOOKBACK_STEPS:-12}" \
  --patch-steps "${PATCH_STEPS:-3}" \
  --epochs "${EPOCHS:-12}" \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --d-model "${D_MODEL:-32}" \
  --layers "${LAYERS:-2}" \
  --heads "${HEADS:-4}" \
  --dropout "${DROPOUT:-0.1}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --entity-aggregation-profile "${ENTITY_AGGREGATION_PROFILE:-mean}"
