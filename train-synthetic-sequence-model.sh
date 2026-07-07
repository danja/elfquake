#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv}"
OUT="${OUT:-data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.torch_sequence.json}"
SUMMARY="${SUMMARY:-data/derived/models/mountain_256x256_seeds40-42_20000.sequence_model_run_summary.json}"
GROUP_PREFIX="${GROUP_PREFIX:-data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.torch_sequence_group}"

manifest_args=()
for seed in 40 41 42; do
  manifest_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_avalanche_sequence/manifest.json")
  manifest_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_piezo_sequence/manifest.json")
  manifest_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_summary_sequence/manifest.json")
done

common_args=(
  --input "$INPUT"
  "${manifest_args[@]}"
  --lookback-steps "${LOOKBACK_STEPS:-60}"
  --epochs "${EPOCHS:-20}"
  --learning-rate "${LEARNING_RATE:-0.001}"
  --hidden-units "${HIDDEN_UNITS:-24}"
  --batch-size "${BATCH_SIZE:-64}"
  --seed "${SEED:-42}"
)

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli train-torch-sequence-holdout \
  "${common_args[@]}" \
  --out "$OUT" \
  --train-fraction 0.8

group_reports=()
for TEST_GROUP in seed40 seed41 seed42; do
  GROUP_OUT="${GROUP_PREFIX}_${TEST_GROUP}.json"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli train-torch-sequence-group-holdout \
    "${common_args[@]}" \
    --out "$GROUP_OUT" \
    --test-group "$TEST_GROUP"
  group_reports+=(--report "$GROUP_OUT")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-run-reports \
  --report "$OUT" \
  "${group_reports[@]}" \
  --out "$SUMMARY"
