#!/usr/bin/env bash
set -euo pipefail

# Matched generic-model comparison for one dynamics profile. Both runs share
# episodes, targets, folds, seed, and hyperparameters; only damage channels vary.
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TARGET="${TARGET:?set TARGET to a synthetic step-target CSV}"
ROOT="${ROOT:?set ROOT to the profile output directory}"
SEEDS="${SEEDS:?set SEEDS to the profile simulation seeds}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-3000}"
RUN_SEEDS="${RUN_SEEDS:-42}"

mkdir -p "$ROOT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli prepare-transformer-target-input \
  --input "$TARGET" --out "$ROOT/transformer_input.csv" --report "$ROOT/transformer_input.json" \
  --target-field eventlist_target_occurred --target-status-field eventlist_target_status

manifests=()
for seed in $SEEDS; do
  manifests+=(--piezo-sequence-manifest "data/derived/models/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}_piezo_sequence/manifest.json")
done
model_seeds=()
for seed in $RUN_SEEDS; do
  model_seeds+=(--seed "$seed")
done
common=(--target "$ROOT/transformer_input.csv" "${manifests[@]}" "${model_seeds[@]}"
  --lookback-steps "${LOOKBACK_STEPS:-12}" --patch-steps "${PATCH_STEPS:-3}"
  --epochs "${EPOCHS:-4}" --learning-rate "${LEARNING_RATE:-0.001}"
  --d-model "${D_MODEL:-32}" --layers "${LAYERS:-2}" --heads "${HEADS:-4}"
  --dropout "${DROPOUT:-0.1}" --batch-size "${BATCH_SIZE:-32}")

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-piezo-group-holdout \
  "${common[@]}" --out "$ROOT/no_damage.json" \
  --exclude-piezo-field damage_total --exclude-piezo-field damage_max \
  --exclude-piezo-field damage_active_cell_count
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-piezo-group-holdout \
  "${common[@]}" --out "$ROOT/with_damage.json"
