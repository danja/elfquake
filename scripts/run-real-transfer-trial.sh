#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}"
OUT_DIR="${OUT_DIR:-data/derived/models/real_transfer_trial}"
MAP_OUT="${MAP_OUT:-data/derived/maps/real_transfer_trial_heldout_week.png}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli run-real-transfer-trial \
  --real-events "$REAL_EVENTS" \
  --synthetic-events data/derived/sim/mountain_256x256_seed40_20000.avalanche_events.csv \
  --synthetic-events data/derived/sim/mountain_256x256_seed41_20000.avalanche_events.csv \
  --synthetic-events data/derived/sim/mountain_256x256_seed42_20000.avalanche_events.csv \
  --out-dir "$OUT_DIR" \
  --magnitude-threshold "${MAGNITUDE_THRESHOLD:-2.5}" \
  --horizon-days "${HORIZON_DAYS:-7}" \
  --cell-degrees "${CELL_DEGREES:-1.5}" \
  --train-fraction 0.8 \
  --pretrain-epochs "${PRETRAIN_EPOCHS:-30}" \
  --finetune-epochs "${FINETUNE_EPOCHS:-80}" \
  --seed "${SEED:-42}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli render-transfer-trial-map \
  --actual-events "$OUT_DIR/heldout_week_actual_events.csv" \
  --predictions "$OUT_DIR/heldout_week_predictions.csv" \
  --out "$MAP_OUT" \
  --metadata-out "${MAP_OUT%.png}.json"
