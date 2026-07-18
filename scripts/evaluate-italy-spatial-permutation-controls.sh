#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/all_italy_spatial_vlf_image_windows_aligned_windows.csv}"
OUT_DIR="${OUT_DIR:-data/derived/models/all_italy_spatial_permutation_controls}"
SEEDS="${SEEDS:-101 202 303 404 505}"

if [[ ! -f "$INPUT" ]]; then ./scripts/prepare-italy-spatial-model-inputs.sh; fi
mkdir -p "$OUT_DIR"

for SEED in $SEEDS; do
  PERMUTED="$OUT_DIR/permuted_${SEED}.csv"
  REPORT="$OUT_DIR/holdout_${SEED}.json"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli permute-spatial-targets \
    --input "$INPUT" --out "$PERMUTED" --seed "$SEED"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-temporal-holdout \
    --input "$PERMUTED" --out "$REPORT" --train-fraction 0.8 --epochs "${EPOCHS:-100}" \
    --learning-rate "${LEARNING_RATE:-0.2}" --group-by-time
done

printf 'permutation reports: %s\n' "$OUT_DIR"
