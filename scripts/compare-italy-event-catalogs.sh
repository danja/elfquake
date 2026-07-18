#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}"
OUT="${OUT:-data/derived/reports/italy_event_catalog_alignment.json}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli compare-event-catalogs \
  --real-events "$REAL_EVENTS" \
  --synthetic-events data/derived/sim/mountain_256x256_seed40_20000.avalanche_events.csv \
  --synthetic-events data/derived/sim/mountain_256x256_seed41_20000.avalanche_events.csv \
  --synthetic-events data/derived/sim/mountain_256x256_seed42_20000.avalanche_events.csv \
  --out "$OUT" \
  --cell-degrees "${CELL_DEGREES:-1.5}"
