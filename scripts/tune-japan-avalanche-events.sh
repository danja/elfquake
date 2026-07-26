#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-20000}"
SEED="${SEED:-40}"
PREFIX="data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${SEED}_${STEPS}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/japan/events_japan_all.normalized.csv}"
OUT="${OUT:-data/derived/reports/japan-avalanche-event-tuning-reduced.csv}"
WORK_DIR="${WORK_DIR:-${OUT%.csv}}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli tune-avalanche-event-extraction \
  --real-events "$REAL_EVENTS" \
  --avalanche "${PREFIX}.avalanche_signal.csv" \
  --activity "${PREFIX}.avalanche_activity.csv" \
  --grid-width "$WIDTH" \
  --grid-height "$HEIGHT" \
  --quantile 0.975 \
  --quantile 0.99 \
  --local-max-window 120 \
  --local-max-window 480 \
  --max-events 0 \
  --max-events 25 \
  --max-events 50 \
  --out "$OUT" \
  --work-dir "$WORK_DIR"
