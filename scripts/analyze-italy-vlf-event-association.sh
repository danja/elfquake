#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EVENTS="${EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}"
ANOMALY="${ANOMALY:-data/derived/models/self_supervised/real_vlf_anomaly_scores.csv}"
OUT="${OUT:-data/derived/reports/italy_vlf_event_association.json}"
WEEKLY_OUT="${WEEKLY_OUT:-data/derived/reports/italy_vlf_event_association_weekly.csv}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-vlf-event-association-report \
  --events "$EVENTS" --anomaly-scores "$ANOMALY" --out "$OUT" \
  --weekly-out "$WEEKLY_OUT" --permutations "${PERMUTATIONS:-2000}" \
  --seed "${SEED:-42}"
