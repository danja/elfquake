#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv}"
OUT="${OUT:-data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.torch_tabular.json}"
SUMMARY="${SUMMARY:-data/derived/models/mountain_256x256_seeds40-42_20000.model_run_summary.json}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli train-torch-tabular-holdout \
  --input "$INPUT" \
  --out "$OUT" \
  --train-fraction 0.8 \
  --epochs "${EPOCHS:-80}" \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --hidden-units "${HIDDEN_UNITS:-32}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --seed "${SEED:-42}"

for TEST_GROUP in seed40 seed41 seed42; do
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli train-torch-tabular-group-holdout \
    --input "$INPUT" \
    --out "data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.torch_tabular_group_${TEST_GROUP}.json" \
    --test-group "$TEST_GROUP" \
    --epochs "${EPOCHS:-80}" \
    --learning-rate "${LEARNING_RATE:-0.001}" \
    --hidden-units "${HIDDEN_UNITS:-32}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --seed "${SEED:-42}"
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-run-reports \
  --report data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.ablation_smoke.json \
  --report data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.temporal_holdout.json \
  --report data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.group_holdout_seed40.json \
  --report data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.group_holdout_seed41.json \
  --report data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.group_holdout_seed42.json \
  --report "$OUT" \
  --report data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.torch_tabular_group_seed40.json \
  --report data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.torch_tabular_group_seed41.json \
  --report data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.torch_tabular_group_seed42.json \
  --out "$SUMMARY"
