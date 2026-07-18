#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/ingv/events_central_italy_all_available.combined.normalized.csv}"
CELL_DEGREES="${CELL_DEGREES:-1.5}"
OUT_DIR="${OUT_DIR:-data/derived/reports/central_italy_episode_alignment}"

# Durations include the full simulation horizon, including quiet tails.
EVENT_FILES=(
  data/derived/sim/mountain_256x256_seed40_20000.avalanche_events.csv
  data/derived/sim/mountain_256x256_seed41_20000.avalanche_events.csv
  data/derived/sim/mountain_256x256_seed42_20000.avalanche_events.csv
  data/derived/sim/mountain_256x256_seed4300_20000.avalanche_events.csv
  data/derived/sim/mountain_256x256_seed4500_40000.avalanche_events.csv
)
DURATION_DAYS=(13.888888889 13.888888889 13.888888889 13.888888889 27.777777778)

if [[ "${#EVENT_FILES[@]}" -ne "${#DURATION_DAYS[@]}" ]]; then
  echo "episode file and duration lists must have the same length" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
for index in "${!EVENT_FILES[@]}"; do
  input="${EVENT_FILES[$index]}"
  duration="${DURATION_DAYS[$index]}"
  stem="$(basename "$input" .csv)"
  output="$OUT_DIR/${stem}.json"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli compare-event-catalogs \
    --real-events "$REAL_EVENTS" \
    --synthetic-events "$input" \
    --synthetic-duration-days "$duration" \
    --cell-degrees "$CELL_DEGREES" \
    --out "$output"
done

printf 'reports: %s\n' "$OUT_DIR"
