#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv}"
ROOT="${ROOT:-data/derived/models/missing_modality}"
TEST_GROUP="${TEST_GROUP:-seed42}"
LOOKBACK_STEPS="${LOOKBACK_STEPS:-60}"
EPOCHS="${EPOCHS:-8}"
HIDDEN_UNITS="${HIDDEN_UNITS:-24}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SEED="${SEED:-42}"

mkdir -p "$ROOT"

run_case() {
  local name="$1"
  shift
  local manifest_args=("$@")
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli train-torch-sequence-group-holdout \
    --input "$INPUT" \
    "${manifest_args[@]}" \
    --out "$ROOT/${name}_${TEST_GROUP}.json" \
    --test-group "$TEST_GROUP" \
    --lookback-steps "$LOOKBACK_STEPS" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --hidden-units "$HIDDEN_UNITS" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED"
}

piezo_args=()
no_piezo_args=()
for seed in 40 41 42; do
  piezo_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_piezo_sequence/manifest.json")
  no_piezo_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_avalanche_sequence/manifest.json")
  no_piezo_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_summary_sequence/manifest.json")
done

run_case "piezo_only" "${piezo_args[@]}"
run_case "no_piezo" "${no_piezo_args[@]}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-run-reports \
  --report "$ROOT/piezo_only_${TEST_GROUP}.json" \
  --report "$ROOT/no_piezo_${TEST_GROUP}.json" \
  --out "$ROOT/missing_modality_${TEST_GROUP}_summary.json"
