#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SOURCE_INPUT="${SOURCE_INPUT:-data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv}"
SPLIT_ROOT="${SPLIT_ROOT:-data/derived/models/sequence_full_balanced}"
ROOT="${ROOT:-data/derived/models/tiny_patch_transformer}"
REGIME_INPUT="${REGIME_INPUT:-$SPLIT_ROOT/mountain_256x256_seeds40-42_20000.post_burn_in_regimes.csv}"
REGIME_REPORT="${REGIME_REPORT:-$SPLIT_ROOT/synthetic_regimes.json}"
SPLIT_INPUT="${SPLIT_INPUT:-$SPLIT_ROOT/mountain_256x256_seeds40-42_20000.regime_balanced_split.csv}"
SPLIT_REPORT="${SPLIT_REPORT:-$SPLIT_ROOT/regime_balanced_split.json}"
OUT="${OUT:-$ROOT/tiny_patch_transformer_balanced.json}"
SUMMARY="${SUMMARY:-$ROOT/tiny_patch_transformer_model_run_summary.json}"
REGIME_COUNT="${REGIME_COUNT:-5}"
BURN_IN_FRACTION="${BURN_IN_FRACTION:-0.2}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"
LOOKBACK_STEPS="${LOOKBACK_STEPS:-60}"
PATCH_STEPS="${PATCH_STEPS:-10}"
D_MODEL="${D_MODEL:-32}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-2}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"

manifest_args=()
for seed in 40 41 42; do
  manifest_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_avalanche_sequence/manifest.json")
  manifest_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_piezo_sequence/manifest.json")
  manifest_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_summary_sequence/manifest.json")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli annotate-synthetic-regimes \
  --input "$SOURCE_INPUT" \
  --out "$REGIME_INPUT" \
  --report "$REGIME_REPORT" \
  --regime-count "$REGIME_COUNT" \
  --burn-in-fraction "$BURN_IN_FRACTION" \
  --drop-burn-in

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli assign-balanced-split \
  --input "$REGIME_INPUT" \
  --out "$SPLIT_INPUT" \
  --report "$SPLIT_REPORT" \
  --group-field synthetic_regime_id \
  --test-fraction "$TEST_FRACTION"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli train-torch-patch-transformer-split-holdout \
  --input "$SPLIT_INPUT" \
  "${manifest_args[@]}" \
  --out "$OUT" \
  --lookback-steps "$LOOKBACK_STEPS" \
  --patch-steps "$PATCH_STEPS" \
  --epochs "$EPOCHS" \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --d-model "$D_MODEL" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --dropout "$DROPOUT" \
  --batch-size "$BATCH_SIZE" \
  --seed "${SEED:-42}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-run-reports \
  --report "$OUT" \
  --out "$SUMMARY"
