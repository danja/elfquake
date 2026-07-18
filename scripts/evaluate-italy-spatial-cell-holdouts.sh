#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/all_italy_spatial_vlf_image_windows_aligned_windows.csv}"
OUT_DIR="${OUT_DIR:-data/derived/models/all_italy_spatial_cell_holdouts_v2}"
EPOCHS="${EPOCHS:-100}"

if [[ ! -f "$INPUT" ]]; then
  ./scripts/prepare-italy-spatial-model-inputs.sh
fi
mkdir -p "$OUT_DIR"

mapfile -t CELLS < <(PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -c '
import csv, sys
values = sorted({row["target_cell_id"] for row in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"))})
print("\n".join(values))
' "$INPUT")

for CELL in "${CELLS[@]}"; do
  SAFE_CELL="${CELL//./_}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-group-holdout \
    --input "$INPUT" \
    --out "$OUT_DIR/cell_${SAFE_CELL}.json" \
    --group-field target_cell_id \
    --test-group "$CELL" \
    --epochs "$EPOCHS" \
    --learning-rate "${LEARNING_RATE:-0.2}"
done

printf 'cell holdout reports: %s (%s cells)\n' "$OUT_DIR" "${#CELLS[@]}"
