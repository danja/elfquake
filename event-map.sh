#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./event-map.sh [events_csv] [output_png] [metadata_json]

Defaults:
  events_csv     newest synthetic event CSV, else newest combined normalized INGV CSV
  output_png     data/derived/maps/<input-stem>.event_map.png
  metadata_json  same prefix with .json

Environment:
  MIN_MAGNITUDE  optional minimum magnitude filter
  MAX_EVENTS     optional maximum number of latest rows to plot
  TITLE          optional map title
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" ]]; then
  input="$1"
else
  input="$(find data/derived/sim -maxdepth 1 -type f -name '*.synthetic_events.csv' -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1s/^[^ ]* //p')"
  if [[ -z "$input" ]]; then
    input="$(find data/derived/ingv -maxdepth 1 -type f -name '*.combined.normalized.csv' -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1s/^[^ ]* //p')"
  fi
fi

if [[ -z "$input" || ! -f "$input" ]]; then
  echo "error: no event CSV found. Build a synthetic event list, or pass a normalized event CSV explicitly." >&2
  exit 2
fi

mkdir -p data/derived/maps
stem="$(basename "$input" .csv)"
output="${2:-data/derived/maps/${stem}.event_map.png}"
metadata="${3:-${output%.png}.json}"
title="${TITLE:-ELFQuake synthetic event map}"

args=(
  -m elfquake.cli render-event-map
  --events "$input"
  --out "$output"
  --metadata-out "$metadata"
  --title "$title"
)

if [[ "${MIN_MAGNITUDE:-}" ]]; then
  args+=(--min-magnitude "$MIN_MAGNITUDE")
fi
if [[ "${MAX_EVENTS:-}" ]]; then
  args+=(--max-events "$MAX_EVENTS")
fi

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python "${args[@]}"
