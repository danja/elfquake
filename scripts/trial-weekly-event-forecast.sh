#!/usr/bin/env bash
set -euo pipefail

. "$(dirname "$0")/lib/staleness.sh"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ROOT:-data/derived/models/trial_forecast}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}"
OUT="${OUT:-$ROOT/mag_gt2_weekly_trial_forecast.json}"
EVENTS_OUT="${EVENTS_OUT:-$ROOT/mag_gt2_weekly_trial_events.csv}"

# As-of date from the catalog's own recorded coverage end, not a literal. This
# default used to be hardcoded at 2026-07-08, so every run forecast the same
# July week no matter how far the catalog had advanced -- a refreshed input
# feeding a frozen question, which is the item-89 pattern in the argument list
# rather than in a file path. Set AS_OF_UTC explicitly to reproduce an old run.
AS_OF_UTC="${AS_OF_UTC:-$(catalog_coverage_end "${FRESHNESS:-data/derived/ingv/catalog_freshness.json}" "2026-07-08T00:00:00Z")}"

# The VLF windows and the anomaly forecast are model artifacts that no refresh
# rebuilds; the event catalog is rewritten every 30 minutes.
require_fresh_inputs "$REAL_EVENTS" \
  "${VLF_WINDOW_ALL:-data/derived/models/all_italy.real_vlf_aligned_windows.csv}" \
  "${VLF_WINDOW_CENTRAL:-data/derived/models/central_italy.real_vlf_aligned_windows.csv}" \
  "${VLF_ANOMALY_REPORT:-data/derived/models/self_supervised/real_vlf_anomaly_forecast.json}"

mkdir -p "$ROOT"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli generate-trial-weekly-event-forecast \
  --real-events "$REAL_EVENTS" \
  --out "$OUT" \
  --events-out "$EVENTS_OUT" \
  --as-of-utc "$AS_OF_UTC" \
  --horizon-days "${HORIZON_DAYS:-7}" \
  --magnitude-threshold "${MAGNITUDE_THRESHOLD:-2.0}" \
  --max-events "${MAX_EVENTS:-25}" \
  --seed "${SEED:-42}" \
  --synthetic-event-glob "${SYNTHETIC_EVENT_GLOB_1:-data/derived/sim/*.synthetic_events.csv}" \
  --synthetic-event-glob "${SYNTHETIC_EVENT_GLOB_2:-data/derived/sim/*.avalanche_events.csv}" \
  --vlf-window "${VLF_WINDOW_ALL:-data/derived/models/all_italy.real_vlf_aligned_windows.csv}" \
  --vlf-window "${VLF_WINDOW_CENTRAL:-data/derived/models/central_italy.real_vlf_aligned_windows.csv}" \
  --vlf-anomaly-report "${VLF_ANOMALY_REPORT:-data/derived/models/self_supervised/real_vlf_anomaly_forecast.json}" \
  --vlf-audio-glob "${VLF_AUDIO_GLOB:-data/derived/vlf/*.audio_features.csv}" \
  --astronomy-glob "${ASTRONOMY_GLOB:-data/raw/astronomy/captures/**/*.json}"
