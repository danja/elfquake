#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
START="${START:-2026-06-29T00:00:00Z}"
AS_OF="${AS_OF:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
END="${END:-$AS_OF}"
COMBINE_START_DATE="${COMBINE_START_DATE:-2026-06-01}"
LOOKBACK_HOURS="${LOOKBACK_HOURS:-24}"
HORIZON_DAYS="${HORIZON_DAYS:-7}"
TARGET_MAGNITUDE_MIN="${TARGET_MAGNITUDE_MIN:-3.0}"
MIN_ANCHOR_GAP_SECONDS="${MIN_ANCHOR_GAP_SECONDS:-60}"
VLF_METADATA_ROOT="${VLF_METADATA_ROOT:-data/raw/vlf/cumiana/captures}"
SPACE_WEATHER_ROOT="${SPACE_WEATHER_ROOT:-data/derived/astronomy}"
START_DATE="${START:0:10}"
END_DATE="${END:0:10}"

# Fetch, normalize, and combine INGV events. Extracted to its own script so the
# live `elfquake-prospective.service` runs exactly this code every 30 minutes
# instead of leaving the catalog frozen between manual runs (see MISTAKES.md,
# 2026-08-21). `START` is passed through explicitly here, so this stays a
# full-window refresh rather than the service's rolling one.
START="$START" END="$END" AS_OF="$AS_OF" COMBINE_START_DATE="$COMBINE_START_DATE" \
ALL_ITALY_EVENTS="${ALL_ITALY_EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}" \
ALL_CENTRAL_EVENTS="${ALL_CENTRAL_EVENTS:-data/derived/ingv/events_central_italy_all_available.combined.normalized.csv}" \
  ./scripts/refresh-ingv-events.sh

ITALY_CURRENT_EVENTS="data/derived/ingv/events_italy_prospective.current.normalized.csv"
CENTRAL_CURRENT_EVENTS="data/derived/ingv/events_central_italy_prospective.current.normalized.csv"

VLF_IMAGE_FEATURES="data/derived/multimodal/cumiana_last_E_VLF.image_features.csv"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli extract-vlf-image-features \
  --image-root "$VLF_METADATA_ROOT" \
  --filename-prefix last_E_VLF \
  --out "$VLF_IMAGE_FEATURES"

for scope in all_italy central_italy; do
  if [[ "$scope" == "all_italy" ]]; then
    events="$ITALY_CURRENT_EVENTS"
  else
    events="$CENTRAL_CURRENT_EVENTS"
  fi
  table="data/derived/multimodal/${scope}.prospective_vlf_windows.csv"
  image_table="data/derived/multimodal/${scope}.prospective_vlf_image_windows.csv"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli update-prospective-vlf-table \
    --table "$table" \
    --events "$events" \
    --vlf-metadata-root "$VLF_METADATA_ROOT" \
    --space-weather-root "$SPACE_WEATHER_ROOT" \
    --region-id "$scope" \
    --lookback-hours "$LOOKBACK_HOURS" \
    --horizon-days "$HORIZON_DAYS" \
    --min-anchor-gap-seconds "$MIN_ANCHOR_GAP_SECONDS" \
    --target-magnitude-min "$TARGET_MAGNITUDE_MIN" \
    --out "$table"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli join-vlf-image-features \
    --windows "$table" \
    --image-features "$VLF_IMAGE_FEATURES" \
    --out "$image_table"
done

# Coverage end from the freshness report `refresh-ingv-events.sh` just wrote,
# not `$END`. On a successful fetch these are the same value; on a failed one
# `$END` would assert coverage the catalog does not have, which is the defect
# that put 22 events' worth of false negatives into a held-out partition on
# 2026-08-21 (see MISTAKES.md).
CATALOG_END="$(FRESHNESS=data/derived/ingv/catalog_freshness.json FALLBACK="$END" "$PYTHON_BIN" - <<'PYEOF'
import json, os
from pathlib import Path

report = Path(os.environ["FRESHNESS"])
coverage = ""
if report.is_file():
    coverage = json.loads(report.read_text(encoding="utf-8")).get("coverage_end_utc", "")
print(coverage or os.environ["FALLBACK"])
PYEOF
)"
printf 'labeling against catalog coverage end: %s\n' "$CATALOG_END"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli label-multimodal-targets \
  --input data/derived/multimodal/all_italy.prospective_vlf_image_windows.csv \
  --events "$ITALY_CURRENT_EVENTS" \
  --as-of "$AS_OF" \
  --catalog-end "$CATALOG_END" \
  --out data/derived/multimodal/all_italy.prospective_vlf_image_windows.labeled.csv

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli label-multimodal-targets \
  --input data/derived/multimodal/central_italy.prospective_vlf_image_windows.csv \
  --events "$CENTRAL_CURRENT_EVENTS" \
  --as-of "$AS_OF" \
  --catalog-end "$CATALOG_END" \
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
