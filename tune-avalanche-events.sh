#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./tune-avalanche-events.sh

Ranks direct avalanche event-extraction settings against the extended
central-Italy INGV event shape.

Environment:
  WIDTH        default 256
  HEIGHT       default 256
  STEPS        default 20000
  SEED         default 42
  REAL_EVENTS  default data/derived/ingv/events_central_italy_all_available.combined.normalized.csv
  OUT          default data/derived/sim/<prefix>.central_italy_avalanche_event_tuning_refined.csv
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

width="${WIDTH:-256}"
height="${HEIGHT:-256}"
steps="${STEPS:-20000}"
seed="${SEED:-42}"
prefix="data/derived/sim/mountain_${width}x${height}_seed${seed}_${steps}"
real_events="${REAL_EVENTS:-data/derived/ingv/events_central_italy_all_available.combined.normalized.csv}"
out="${OUT:-${prefix}.central_italy_avalanche_event_tuning_refined.csv}"
work_dir="${WORK_DIR:-${out%.csv}}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli tune-avalanche-event-extraction \
  --real-events "$real_events" \
  --avalanche "${prefix}.avalanche_signal.csv" \
  --activity "${prefix}.avalanche_activity.csv" \
  --grid-width "$width" \
  --grid-height "$height" \
  --quantile 0.999 \
  --quantile 0.9995 \
  --local-max-window 240 \
  --local-max-window 480 \
  --max-events 0 \
  --max-events 5 \
  --max-events 10 \
  --out "$out" \
  --work-dir "$work_dir"
