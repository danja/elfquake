#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv}"
OUT="${OUT:-data/derived/models/mountain_256x256_seeds40-42_20000.synthetic_event_list_targets_h6.csv}"
REPORT="${REPORT:-data/derived/models/mountain_256x256_seeds40-42_20000.synthetic_event_list_targets_h6.json}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-synthetic-event-list-targets \
  --input "$INPUT" \
  --out "$OUT" \
  --report "$REPORT" \
  --horizon-rows "${HORIZON_ROWS:-6}" \
  --magnitude-threshold "${MAGNITUDE_THRESHOLD:-2.0}"
