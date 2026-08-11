#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/all_italy_spatial_vlf_image_windows_aligned_windows.csv}"
OUT_DIR="${OUT_DIR:-data/derived/reports/italy_capture_era_shift}"
OUT="${OUT:-$OUT_DIR/report.json}"
CSV_OUT="${CSV_OUT:-$OUT_DIR/feature_comparisons.csv}"
ERA_CSV_DIR="${ERA_CSV_DIR:-$OUT_DIR/eras}"

if [[ ! -f "$INPUT" ]]; then
  ./scripts/prepare-italy-spatial-model-inputs.sh
fi

mkdir -p "$OUT_DIR"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli diagnose-capture-era-shift \
  --input "$INPUT" \
  --out "$OUT" \
  --csv-out "$CSV_OUT" \
  --era-csv-dir "$ERA_CSV_DIR" \
  --time-field "${TIME_FIELD:-window_start_utc}" \
  --era-gap-hours "${ERA_GAP_HOURS:-48}" \
  --min-era-anchors "${MIN_ERA_ANCHORS:-5}"

printf 'capture era shift report: %s\n' "$OUT"
