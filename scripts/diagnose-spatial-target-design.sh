#!/usr/bin/env bash
# Measure what a candidate fixed-cell target design can express, before any
# model is fitted (next-actions item 103).
#
# The anchors are 30 minutes apart and the current target horizon is 7 days, so
# consecutive target windows overlap by more than 99% and a held-out partition
# shorter than the horizon carries almost no within-cell label variation. Row
# counts therefore overstate the evidence by orders of magnitude. This reports
# the honest quantity instead: how many cells carry both classes, and how many
# times a cell's label changes between consecutive anchors, both over the whole
# record and inside the same grouped-time held-out partition the evaluator uses.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/multimodal/all_italy.prospective_vlf_image_windows.csv}"
EVENTS="${EVENTS:-data/derived/ingv/events_italy_prospective.current.normalized.csv}"
OUT="${OUT:-data/derived/reports/italy_target_design/spatial_target_design.json}"

mkdir -p "$(dirname "$OUT")"

args=(--input "$INPUT" --events "$EVENTS" --out "$OUT" --train-fraction "${TRAIN_FRACTION:-0.8}")
for horizon in ${HORIZON_DAYS:-1 2 3 7}; do args+=(--horizon-days "$horizon"); done
for degrees in ${CELL_DEGREES:-1.5}; do args+=(--cell-degrees "$degrees"); done
for magnitude in ${TARGET_MAGNITUDE_MIN:-2.5}; do args+=(--target-magnitude-min "$magnitude"); done
[[ -n "${CATALOG_END:-}" ]] && args+=(--catalog-end "$CATALOG_END")

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli diagnose-spatial-target-design "${args[@]}"

printf 'target design diagnostic: %s\n' "$OUT"
