#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
START="${START:-2026-06-29T00:00:00Z}"
END="${END:-2026-07-08T00:00:00Z}"
AS_OF="${AS_OF:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
START_DATE="${START:0:10}"
END_DATE="${END:0:10}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli fetch-ingv-events \
  --start "$START" \
  --end "$END"

RAW="$(find data/raw/ingv -maxdepth 1 -name "events_italy_${START_DATE}_${END_DATE}_*.txt" | sort | tail -n 1)"
if [[ -z "$RAW" ]]; then
  echo "No fetched INGV raw file found for ${START_DATE} to ${END_DATE}" >&2
  exit 2
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli normalize-ingv-events \
  --raw "$RAW" \
  --out "data/derived/ingv/events_italy_${START_DATE}_${END_DATE}.normalized.csv"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli normalize-ingv-events \
  --raw "$RAW" \
  --out "data/derived/ingv/events_central_italy_${START_DATE}_${END_DATE}.normalized.csv" \
  --only-region central_italy

italy_inputs=()
while IFS= read -r path; do
  italy_inputs+=(--input "$path")
done < <(find data/derived/ingv -maxdepth 1 -name "events_italy_*.normalized.csv" ! -name "*.combined.normalized.csv" | sort)

central_inputs=()
while IFS= read -r path; do
  central_inputs+=(--input "$path")
done < <(find data/derived/ingv -maxdepth 1 -name "events_central_italy_*.normalized.csv" ! -name "*.combined.normalized.csv" | sort)

ITALY_EVENTS="data/derived/ingv/events_italy_2026-06-01_${END_DATE}.combined.normalized.csv"
CENTRAL_EVENTS="data/derived/ingv/events_central_italy_2026-06-01_${END_DATE}.combined.normalized.csv"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli combine-normalized-events \
  "${italy_inputs[@]}" \
  --out "$ITALY_EVENTS"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli combine-normalized-events \
  "${central_inputs[@]}" \
  --out "$CENTRAL_EVENTS"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli label-multimodal-targets \
  --input data/derived/multimodal/all_italy.prospective_vlf_image_windows.csv \
  --events "$ITALY_EVENTS" \
  --as-of "$AS_OF" \
  --out data/derived/multimodal/all_italy.prospective_vlf_image_windows.labeled.csv

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli label-multimodal-targets \
  --input data/derived/multimodal/central_italy.prospective_vlf_image_windows.csv \
  --events "$CENTRAL_EVENTS" \
  --as-of "$AS_OF" \
  --out data/derived/multimodal/central_italy.prospective_vlf_image_windows.labeled.csv

for scope in all_italy central_italy; do
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-prospective-table \
    --input "data/derived/multimodal/${scope}.prospective_vlf_image_windows.labeled.csv" \
    --as-of "$AS_OF" \
    --out "data/derived/multimodal/${scope}.prospective_vlf_image_windows.labeled.summary.json"

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-readiness \
    --input "data/derived/multimodal/${scope}.prospective_vlf_image_windows.labeled.csv" \
    --out "data/derived/multimodal/${scope}.prospective_vlf_image_windows.labeled.readiness.json"
done
