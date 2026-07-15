#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EVENTS="${EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}"
VLF_ROOT="${VLF_ROOT:-data/raw/vlf/cumiana/captures}"
ANOMALY="${ANOMALY:-data/derived/models/self_supervised/real_vlf_anomaly_scores.csv}"
OUT="${OUT:-data/derived/reports/italy_data_coverage.json}"
WEEKLY_OUT="${WEEKLY_OUT:-data/derived/reports/italy_data_coverage_weekly.csv}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-italy-coverage-report \
  --events "$EVENTS" --vlf-metadata-root "$VLF_ROOT" --anomaly-scores "$ANOMALY" \
  --out "$OUT" --weekly-out "$WEEKLY_OUT"
