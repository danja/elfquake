#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
IMAGE_ROOT="${IMAGE_ROOT:-data/raw/vlf/cumiana/captures}"
OUT_DIR="${OUT_DIR:-data/derived/reports/italy_vlf_palette_shift}"
OUT="${OUT:-$OUT_DIR/report.json}"
CAPTURE_CSV="${CAPTURE_CSV:-$OUT_DIR/captures.csv}"
BAND_CSV="${BAND_CSV:-$OUT_DIR/band_comparison.csv}"

if [[ ! -d "$IMAGE_ROOT" ]]; then
  printf 'missing capture root: %s\n' "$IMAGE_ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli diagnose-vlf-palette-shift \
  --image-root "$IMAGE_ROOT" \
  --filename-prefix "${FILENAME_PREFIX:-last_E_VLF}" \
  --out "$OUT" \
  --capture-csv-out "$CAPTURE_CSV" \
  --band-csv-out "$BAND_CSV" \
  --hour-low "${HOUR_LOW:-11}" \
  --hour-high "${HOUR_HIGH:-13}" \
  --residual-limit "${RESIDUAL_LIMIT:-30}"

printf 'vlf palette shift report: %s\n' "$OUT"
