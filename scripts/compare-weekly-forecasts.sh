#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ROOT:-data/derived/models/forecast_comparison}"
BASELINE_REPORT="${BASELINE_REPORT:-data/derived/models/trial_forecast/mag_gt2_weekly_trial_forecast.json}"
BASELINE_EVENTS="${BASELINE_EVENTS:-data/derived/models/trial_forecast/mag_gt2_weekly_trial_events.csv}"
CANDIDATE_REPORT="${CANDIDATE_REPORT:-data/derived/models/learned_forecast/mag_gt2_weekly_learned_forecast.json}"
CANDIDATE_EVENTS="${CANDIDATE_EVENTS:-data/derived/models/learned_forecast/mag_gt2_weekly_learned_events.csv}"
OUT="${OUT:-$ROOT/trial_vs_learned_weekly_forecast.json}"
CSV_OUT="${CSV_OUT:-$ROOT/trial_vs_learned_weekly_forecast.csv}"

mkdir -p "$ROOT"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli compare-weekly-forecasts \
  --baseline-report "$BASELINE_REPORT" \
  --baseline-events "$BASELINE_EVENTS" \
  --candidate-report "$CANDIDATE_REPORT" \
  --candidate-events "$CANDIDATE_EVENTS" \
  --out "$OUT" \
  --csv-out "$CSV_OUT"
