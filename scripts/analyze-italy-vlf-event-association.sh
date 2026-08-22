#!/usr/bin/env bash
set -euo pipefail

. "$(dirname "$0")/lib/staleness.sh"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EVENTS="${EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}"
ANOMALY="${ANOMALY:-data/derived/models/self_supervised/real_vlf_anomaly_scores.csv}"
OUT="${OUT:-data/derived/reports/italy_vlf_event_association.json}"
WEEKLY_OUT="${WEEKLY_OUT:-data/derived/reports/italy_vlf_event_association_weekly.csv}"

# The refresh path rebuilds the event catalog every 30 minutes; nothing
# rebuilds the anomaly scores. Association strength read off a fresh
# catalog and stale scores is not the association strength of today.
require_fresh_inputs "$EVENTS" "$ANOMALY"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-vlf-event-association-report \
  --events "$EVENTS" --anomaly-scores "$ANOMALY" --out "$OUT" \
  --weekly-out "$WEEKLY_OUT" --permutations "${PERMUTATIONS:-2000}" \
  --seed "${SEED:-42}"
