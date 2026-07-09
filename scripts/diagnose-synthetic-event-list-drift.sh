#!/usr/bin/env bash
set -euo pipefail

INPUT="${INPUT:-data/derived/models/mountain_256x256_seeds40-42_20000.synthetic_event_list_targets_h6.csv}"
OUT="${OUT:-data/derived/models/synthetic_event_list_drift/h6_drift.json}"
CSV_OUT="${CSV_OUT:-data/derived/models/synthetic_event_list_drift/h6_drift_buckets.csv}"
TARGET_FIELD="${TARGET_FIELD:-eventlist_target_occurred}"
TARGET_STATUS_FIELD="${TARGET_STATUS_FIELD:-eventlist_target_status}"
GROUP_FIELD="${GROUP_FIELD:-dataset_id}"
TIME_FIELD="${TIME_FIELD:-window_start_utc}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
BUCKET_COUNT="${BUCKET_COUNT:-10}"
TOP_N="${TOP_N:-20}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli diagnose-synthetic-drift \
  --input "$INPUT" \
  --out "$OUT" \
  --csv-out "$CSV_OUT" \
  --target-field "$TARGET_FIELD" \
  --target-status-field "$TARGET_STATUS_FIELD" \
  --group-field "$GROUP_FIELD" \
  --time-field "$TIME_FIELD" \
  --train-fraction "$TRAIN_FRACTION" \
  --bucket-count "$BUCKET_COUNT" \
  --top-n "$TOP_N"
