#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/all_italy_spatial_vlf_image_windows_aligned_windows.csv}"
OUT="${OUT:-data/derived/models/all_italy_spatial_vlf_image_windows.temporal_grouped_holdout.json}"
# Optional. Set to `target_cell_id` to add the item-103 cell-stratified metric
# and the per-cell base-rate control to this run and to anything that wraps it
# (the coordinate control). Empty keeps the historical pooled-only report.
STRATIFY_FIELD="${STRATIFY_FIELD:-}"

STRATIFY_ARGS=()
if [[ -n "$STRATIFY_FIELD" ]]; then STRATIFY_ARGS=(--stratify-field "$STRATIFY_FIELD"); fi

if [[ ! -f "$INPUT" ]]; then
  ./scripts/prepare-italy-spatial-model-inputs.sh
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-temporal-holdout \
  --input "$INPUT" \
  --out "$OUT" \
  --time-field window_start_utc \
  --train-fraction "${TRAIN_FRACTION:-0.8}" \
  --epochs "${EPOCHS:-600}" \
  --learning-rate "${LEARNING_RATE:-0.2}" \
  --group-by-time \
  "${STRATIFY_ARGS[@]}"

printf 'grouped spatial baseline: %s\n' "$OUT"
