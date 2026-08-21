#!/usr/bin/env bash
# Within-cell null for the fixed-cell spatial baseline (next-actions item
# 105(d)). Use this in place of evaluate-italy-spatial-permutation-controls.sh.
#
# The timestamp-shuffle control is not a null at this record length. Target
# windows overlap by more than 99%, so most of what the labels contain is
# autocorrelation, and shuffling time slices converts a few long runs into many
# short ones. On the item-104 tables that took held-out transitions from 1 to
# 94-119 in era_0 and from 19 to 315-342 in era_3 -- the control was scored on a
# task with up to a hundred times the evidence of the run it was meant to null.
#
# A circular shift moves the whole labeled matrix in time by one offset. Within
# every cell the sequence length and positive count are preserved exactly and
# the run structure is preserved apart from the wrap seam, while the alignment
# between features and labels -- the only place a temporal signal could live --
# is destroyed. The printed before/after transition counts are the check that
# this held; they should differ by a couple at most.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/all_italy_spatial_vlf_image_windows_aligned_windows.csv}"
OUT_DIR="${OUT_DIR:-data/derived/models/all_italy_spatial_shift_controls}"
SEEDS="${SEEDS:-101 202 303 404 505}"
# Match the real run's epochs. Item 93 was distorted by comparing 100-epoch
# controls against a 600-epoch run.
EPOCHS="${EPOCHS:-600}"
STRATIFY_FIELD="${STRATIFY_FIELD:-}"

STRATIFY_ARGS=()
if [[ -n "$STRATIFY_FIELD" ]]; then STRATIFY_ARGS=(--stratify-field "$STRATIFY_FIELD"); fi

if [[ ! -f "$INPUT" ]]; then ./scripts/prepare-italy-spatial-model-inputs.sh; fi
mkdir -p "$OUT_DIR"

for SEED in $SEEDS; do
  SHIFTED="$OUT_DIR/shifted_${SEED}.csv"
  REPORT="$OUT_DIR/holdout_${SEED}.json"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli shift-spatial-targets \
    --input "$INPUT" --out "$SHIFTED" --seed "$SEED" \
    --time-field "${TIME_FIELD:-window_start_utc}" \
    --cell-field "${CELL_FIELD:-target_cell_id}" \
    --min-shift-fraction "${MIN_SHIFT_FRACTION:-0.1}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-temporal-holdout \
    --input "$SHIFTED" --out "$REPORT" --train-fraction "${TRAIN_FRACTION:-0.8}" \
    --epochs "$EPOCHS" --learning-rate "${LEARNING_RATE:-0.2}" --group-by-time \
    "${STRATIFY_ARGS[@]}"
done

printf 'shift control reports: %s\n' "$OUT_DIR"
