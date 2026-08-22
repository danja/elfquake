#!/usr/bin/env bash
# Third standing control for the fixed-cell spatial baseline, alongside the
# coordinate control and the permutation control.
#
# Every fixed-cell result so far has reduced to a static per-cell base rate:
# `all_features` is the only ablation above chance and the only one holding the
# cell coordinates, and dropping them collapses it to 0.5 (next-actions items
# 93, 96, 102). Pooled balanced accuracy cannot separate "predicts when an event
# happens" from "knows which cells are seismically active", because a predictor
# that is constant within a cell still scores above chance pooled.
#
# Scoring balanced accuracy inside each cell and averaging over cells removes
# that: any cell-constant predictor scores exactly 0.5. The run also reports an
# explicit `stratum_base_rate` control -- the per-cell training positive rate
# used directly as the score -- so the quantity being controlled for is named
# and measured rather than inferred.
set -euo pipefail

. "$(dirname "$0")/lib/staleness.sh"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/all_italy_spatial_vlf_image_windows_aligned_windows.csv}"
OUT_DIR="${OUT_DIR:-data/derived/reports/italy_cell_stratified}"
OUT="${OUT:-$OUT_DIR/$(basename "${INPUT%.csv}").cell_stratified_holdout.json}"
STRATIFY_FIELD="${STRATIFY_FIELD:-target_cell_id}"

mkdir -p "$(dirname "$OUT")"

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
  --stratify-field "$STRATIFY_FIELD"

printf 'cell-stratified holdout: %s\n' "$OUT"
