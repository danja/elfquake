#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
START="${START:-2026-01-01T00:00:00Z}"
END="${END:-2026-07-08T00:00:00Z}"
CHUNK_DAYS="${CHUNK_DAYS:-14}"
MIN_MAG="${MIN_MAG:-2.0}"
START_DATE="${START:0:10}"
END_DATE="${END:0:10}"
PLAN="${PLAN:-data/derived/backfill/ingv_italy_${START_DATE}_${END_DATE}.plan.csv}"
ALL_EVENTS="${ALL_EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}"
CENTRAL_EVENTS="${CENTRAL_EVENTS:-data/derived/ingv/events_central_italy_all_available.combined.normalized.csv}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli plan-ingv-backfill \
  --start "$START" \
  --end "$END" \
  --chunk-days "$CHUNK_DAYS" \
  --min-mag "$MIN_MAG" \
  --out "$PLAN"

tail -n +2 "$PLAN" | while IFS=, read -r CHUNK_START CHUNK_END _COMMAND; do
  CHUNK_START_DATE="${CHUNK_START:0:10}"
  CHUNK_END_DATE="${CHUNK_END:0:10}"

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli fetch-ingv-events \
    --start "$CHUNK_START" \
    --end "$CHUNK_END" \
    --min-mag "$MIN_MAG"

  RAW="$(find data/raw/ingv -maxdepth 1 -name "events_italy_${CHUNK_START_DATE}_${CHUNK_END_DATE}_*.txt" | sort | tail -n 1)"
  if [[ -z "$RAW" ]]; then
    echo "No fetched INGV raw file found for ${CHUNK_START_DATE} to ${CHUNK_END_DATE}" >&2
    exit 2
  fi

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli normalize-ingv-events \
    --raw "$RAW" \
    --out "data/derived/ingv/events_italy_${CHUNK_START_DATE}_${CHUNK_END_DATE}.normalized.csv"

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli normalize-ingv-events \
    --raw "$RAW" \
    --out "data/derived/ingv/events_central_italy_${CHUNK_START_DATE}_${CHUNK_END_DATE}.normalized.csv" \
    --only-region central_italy
done

italy_inputs=()
while IFS= read -r path; do
  italy_inputs+=(--input "$path")
done < <(find data/derived/ingv -maxdepth 1 -name "events_italy_*.normalized.csv" ! -name "*.combined.normalized.csv" | sort)

central_inputs=()
while IFS= read -r path; do
  central_inputs+=(--input "$path")
done < <(find data/derived/ingv -maxdepth 1 -name "events_central_italy_*.normalized.csv" ! -name "*.combined.normalized.csv" | sort)

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli combine-normalized-events \
  "${italy_inputs[@]}" \
  --out "$ALL_EVENTS"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli combine-normalized-events \
  "${central_inputs[@]}" \
  --out "$CENTRAL_EVENTS"

for scope in all_italy central_italy; do
  if [[ "$scope" == "all_italy" ]]; then
    EVENTS="$ALL_EVENTS"
  else
    EVENTS="$CENTRAL_EVENTS"
  fi
  WINDOWS="data/derived/models/${scope}.ingv_backfill_seismic_windows.csv"
  READINESS="data/derived/models/${scope}.ingv_backfill_seismic_windows.readiness.json"

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-seismic-training-windows \
    --events "$EVENTS" \
    --region-id "$scope" \
    --start "$START" \
    --end "$END" \
    --out "$WINDOWS"

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-readiness \
    --input "$WINDOWS" \
    --out "$READINESS"
done
