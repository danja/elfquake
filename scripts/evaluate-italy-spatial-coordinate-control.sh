#!/usr/bin/env bash
# Control for the fixed-cell spatial baseline: rerun it with the cell coordinate
# columns removed. `target_cell_id` is already an ID field, but the numeric
# latitude/longitude/cell-size columns still enter the design matrix, which lets
# the model learn a static per-cell base rate and report it as skill.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/all_italy_spatial_vlf_image_windows_aligned_windows.csv}"
OUT_DIR="${OUT_DIR:-data/derived/reports/italy_capture_era_shift}"
STRIPPED="${STRIPPED:-$OUT_DIR/$(basename "${INPUT%.csv}").no_cell_coordinates.csv}"
OUT="${OUT:-$OUT_DIR/$(basename "${INPUT%.csv}").no_cell_coordinates.temporal_grouped_holdout.json}"
DROP_FIELDS="${DROP_FIELDS:-target_cell_latitude,target_cell_longitude,target_cell_degrees}"
# Forwarded to the baseline script. `target_cell_id` survives the strip, so the
# cell-stratified metric can still be computed on the control run.
STRATIFY_FIELD="${STRATIFY_FIELD:-}"

mkdir -p "$(dirname "$STRIPPED")" "$(dirname "$OUT")"

INPUT="$INPUT" STRIPPED="$STRIPPED" DROP_FIELDS="$DROP_FIELDS" "$PYTHON_BIN" - <<'PY'
import csv, os
from pathlib import Path

source = Path(os.environ["INPUT"])
target = Path(os.environ["STRIPPED"])
drop = {name for name in os.environ["DROP_FIELDS"].split(",") if name}

with source.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fields = [name for name in (reader.fieldnames or []) if name not in drop]
    rows = [{name: row[name] for name in fields} for row in reader]

with target.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"dropped: {sorted(drop)}")
print(f"stripped rows: {len(rows)} fields: {len(fields)}")
PY

INPUT="$STRIPPED" OUT="$OUT" STRATIFY_FIELD="$STRATIFY_FIELD" \
  ./scripts/evaluate-italy-spatial-baseline.sh

printf 'coordinate control: %s\n' "$OUT"
