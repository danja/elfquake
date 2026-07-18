#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUT="${OUT:-data/derived/sim/mountain_256x256_seeds40-42_20000.central_combined_events.csv}"
REPORT="${REPORT:-${OUT%.csv}.json}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli combine-synthetic-catalogs \
  --synthetic-events data/derived/sim/mountain_256x256_seed40_20000.avalanche_events.csv \
  --synthetic-events data/derived/sim/mountain_256x256_seed41_20000.avalanche_events.csv \
  --synthetic-events data/derived/sim/mountain_256x256_seed42_20000.avalanche_events.csv \
  --out "$OUT" --report "$REPORT" --offset-days "${OFFSET_DAYS:-21}"
