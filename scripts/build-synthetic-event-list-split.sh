#!/usr/bin/env bash
set -euo pipefail

INPUT="${INPUT:-data/derived/models/mountain_256x256_seeds40-42_20000.synthetic_event_list_targets_h6.csv}"
OUT="${OUT:-data/derived/models/mountain_256x256_seeds40-42_20000.synthetic_event_list_targets_h6.balanced_split.csv}"
REPORT="${REPORT:-data/derived/models/mountain_256x256_seeds40-42_20000.synthetic_event_list_targets_h6.balanced_split.json}"
GROUP_FIELD="${GROUP_FIELD:-dataset_id}"
TARGET_FIELD="${TARGET_FIELD:-eventlist_target_occurred}"
SPLIT_FIELD="${SPLIT_FIELD:-model_split}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli assign-balanced-split \
  --input "$INPUT" \
  --out "$OUT" \
  --report "$REPORT" \
  --group-field "$GROUP_FIELD" \
  --target-field "$TARGET_FIELD" \
  --split-field "$SPLIT_FIELD" \
  --test-fraction "$TEST_FRACTION"
