#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}"
SYNTHETIC_EVENTS="${SYNTHETIC_EVENTS:-data/derived/sim/mountain_256x256_seed40_20000.avalanche_events.csv}"
OUT="${OUT:-data/derived/sim/mountain_256x256_seed40_20000.avalanche_events.catalog_calibrated.csv}"
REPORT="${REPORT:-${OUT%.csv}.json}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli calibrate-synthetic-catalog \
  --real-events "$REAL_EVENTS" --synthetic-events "$SYNTHETIC_EVENTS" \
  --out "$OUT" --report "$REPORT" --seed "${SEED:-42}"
