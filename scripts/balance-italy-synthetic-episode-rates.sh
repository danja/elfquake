#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/ingv/events_central_italy_all_available.combined.normalized.csv}"
SYNTHETIC_EVENTS="${SYNTHETIC_EVENTS:-data/derived/sim/mountain_256x256_seeds40-42_4300-4500_combined_events.csv}"
OUT="${OUT:-data/derived/sim/mountain_256x256_seeds40-42_4300-4500.episode_rate_balanced_events.csv}"
REPORT="${REPORT:-data/derived/reports/central_italy_episode_rate_balance.json}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli balance-synthetic-episode-rates \
  --real-events "$REAL_EVENTS" \
  --synthetic-events "$SYNTHETIC_EVENTS" \
  --episode-duration-days 13.888888889 \
  --episode-duration-days 13.888888889 \
  --episode-duration-days 13.888888889 \
  --episode-duration-days 13.888888889 \
  --episode-duration-days 27.777777778 \
  --out "$OUT" \
  --report "$REPORT" \
  --seed "${SEED:-42}"
