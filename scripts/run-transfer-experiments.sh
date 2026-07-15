#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}"
OUT="${OUT:-data/derived/models/real_transfer_trial/experiment-suite.json}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli run-transfer-experiment-suite \
  --real-events "$REAL_EVENTS" \
  --synthetic-events "${SYNTHETIC_EVENT_1:-data/derived/sim/mountain_256x256_seed40_20000.synthetic_events.csv}" \
  --synthetic-events "${SYNTHETIC_EVENT_2:-data/derived/sim/mountain_256x256_seed41_20000.synthetic_events.csv}" \
  --synthetic-events "${SYNTHETIC_EVENT_3:-data/derived/sim/mountain_256x256_seed42_20000.synthetic_events.csv}" \
  --synthetic-events "${SYNTHETIC_EVENT_4:-data/derived/sim/mountain_256x256_seed4300_20000.synthetic_events.csv}" \
  --out "$OUT" \
  --magnitude-threshold "${MAGNITUDE_THRESHOLD:-2.5}" \
  --horizon-days "${HORIZON_DAYS:-7}" \
  --cell-degrees "${CELL_DEGREES:-1.5}" \
  --epochs "${EPOCHS:-50}" \
  --pretrain-epochs "${PRETRAIN_EPOCHS:-30}" \
  --seed "${SEED:-42}"
