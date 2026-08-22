#!/usr/bin/env bash
set -euo pipefail

. "$(dirname "$0")/lib/staleness.sh"

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
else
  # Item 89 replaced the existence-only rebuild guard inside
  # prepare-italy-spatial-model-inputs.sh, but every caller kept its own, so a
  # refreshed labeled table still stopped here: the aligned dataset exists, the
  # branch is skipped, and the evaluation reports a four-day-old record with
  # today's timestamp. Rebuild when missing, warn when merely stale.
  require_fresh_inputs "${SPATIAL_LABELS:-data/derived/multimodal/all_italy.spatial_vlf_image_windows.labeled.csv}" "$INPUT"
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
