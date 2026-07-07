#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv}"
ROOT="${ROOT:-data/derived/models/sequence_full_balanced}"
REGIME_INPUT="${REGIME_INPUT:-$ROOT/mountain_256x256_seeds40-42_20000.post_burn_in_regimes.csv}"
REGIME_REPORT="${REGIME_REPORT:-$ROOT/synthetic_regimes.json}"
SPLIT_INPUT="${SPLIT_INPUT:-$ROOT/mountain_256x256_seeds40-42_20000.regime_balanced_split.csv}"
SPLIT_REPORT="${SPLIT_REPORT:-$ROOT/regime_balanced_split.json}"
OUT="${OUT:-$ROOT/torch_sequence_full_balanced.json}"
SUMMARY="${SUMMARY:-$ROOT/sequence_full_balanced_model_run_summary.json}"
REGIME_COUNT="${REGIME_COUNT:-5}"
BURN_IN_FRACTION="${BURN_IN_FRACTION:-0.2}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"
LOOKBACK_STEPS="${LOOKBACK_STEPS:-60}"
HIDDEN_UNITS="${HIDDEN_UNITS:-24}"
EPOCHS="${EPOCHS:-20}"

manifest_args=()
for seed in 40 41 42; do
  manifest_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_avalanche_sequence/manifest.json")
  manifest_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_piezo_sequence/manifest.json")
  manifest_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_summary_sequence/manifest.json")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli annotate-synthetic-regimes \
  --input "$INPUT" \
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

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli train-torch-sequence-split-holdout \
  --input "$SPLIT_INPUT" \
  "${manifest_args[@]}" \
  --out "$OUT" \
  --lookback-steps "$LOOKBACK_STEPS" \
  --epochs "$EPOCHS" \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --hidden-units "$HIDDEN_UNITS" \
  --batch-size "${BATCH_SIZE:-64}" \
  --seed "${SEED:-42}" \
  --evaluation sequence_full

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-run-reports \
  --report "$OUT" \
  --out "$SUMMARY"
