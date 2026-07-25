#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EVENTS="${EVENTS:-data/derived/japan/events_japan_all.normalized.csv}"
VLF_WINDOWS="${VLF_WINDOWS:-data/derived/japan/japan.vlf_cdf_window_features.csv}"
OUT_DIR="${OUT_DIR:-data/derived/reports/japan_target_thresholds}"
START="${START:-2025-01-01T00:00:00Z}"
END="${END:-2026-07-08T00:00:00Z}"
THRESHOLDS="${THRESHOLDS:-3.0 3.5 4.0 4.5 5.0 5.5}"

mkdir -p "$OUT_DIR"
for threshold in $THRESHOLDS; do
  slug="${threshold//./_}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-seismic-training-windows \
    --events "$EVENTS" --region-id japan --start "$START" --end "$END" \
    --window-days 7 --horizon-days 7 --target-magnitude-min "$threshold" \
    --out "$OUT_DIR/japan.seismic_training_windows.m${slug}.csv" >/dev/null
done

EVENTS="$EVENTS" VLF_WINDOWS="$VLF_WINDOWS" OUT_DIR="$OUT_DIR" THRESHOLDS="$THRESHOLDS" \
  "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
vlf_path = Path(os.environ["VLF_WINDOWS"])
vlf_rows = list(csv.DictReader(vlf_path.open(newline="", encoding="utf-8")))
vlf_ids = {row["window_id"] for row in vlf_rows if int(row.get("japan_vlf_row_count", "0") or 0) > 0}
report = []
for threshold in os.environ["THRESHOLDS"].split():
    slug = threshold.replace(".", "_")
    path = out_dir / f"japan.seismic_training_windows.m{slug}.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    report.append({
        "target_magnitude_min": float(threshold),
        "window_count": len(rows),
        "positive_count": sum(row["target_occurred"] == "1" for row in rows),
        "negative_count": sum(row["target_occurred"] == "0" for row in rows),
        "vlf_overlap_count": sum(row["window_id"] in vlf_ids for row in rows),
    })
(out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
for row in report:
    print("M{target_magnitude_min:g}: {positive_count}/{negative_count} positive/negative, {vlf_overlap_count} VLF-overlap windows".format(**row))
PY
