#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
START="${START:-2024-01-01T00:00:00Z}"
END="${END:-2026-07-08T00:00:00Z}"
CHUNK_DAYS="${CHUNK_DAYS:-14}"
MIN_MAG="${MIN_MAG:-2.0}"
RAW_ROOT="${RAW_ROOT:-data/raw/japan/events}"
DERIVED_ROOT="${DERIVED_ROOT:-data/derived/japan}"
PLAN="${PLAN:-data/derived/backfill/japan_${START:0:10}_${END:0:10}.plan.csv}"
ALL_EVENTS="${ALL_EVENTS:-$DERIVED_ROOT/events_japan_all.normalized.csv}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli plan-ingv-backfill \
  --start "$START" --end "$END" --chunk-days "$CHUNK_DAYS" --min-mag "$MIN_MAG" --out "$PLAN"

tail -n +2 "$PLAN" | while IFS=, read -r CHUNK_START CHUNK_END _COMMAND; do
  START_DATE="${CHUNK_START:0:10}"
  END_DATE="${CHUNK_END:0:10}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli fetch-japan-events \
    --start "$CHUNK_START" --end "$CHUNK_END" --min-mag "$MIN_MAG" --out-root "$RAW_ROOT"
  RAW="$(find "$RAW_ROOT" -maxdepth 1 -name "events_japan_${START_DATE}_${END_DATE}_*.json" | sort | tail -n 1)"
  [[ -n "$RAW" ]] || { echo "No Japan raw file found for $START_DATE to $END_DATE" >&2; exit 2; }
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli normalize-japan-events \
    --raw "$RAW" --out "$DERIVED_ROOT/events_japan_${START_DATE}_${END_DATE}.normalized.csv"
done

inputs=()
while IFS= read -r path; do inputs+=(--input "$path"); done < <(
  find "$DERIVED_ROOT" -maxdepth 1 -name 'events_japan_*.normalized.csv' ! -name '*.combined.normalized.csv' | sort
)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli combine-normalized-events \
  "${inputs[@]}" --out "$ALL_EVENTS"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-seismic-training-windows \
  --events "$ALL_EVENTS" --region-id japan --start "$START" --end "$END" \
  --out "${DERIVED_ROOT}/japan.seismic_training_windows.csv"

echo "Japan events: $ALL_EVENTS"
