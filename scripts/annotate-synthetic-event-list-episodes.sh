#!/usr/bin/env bash
set -euo pipefail

INPUT="${INPUT:-data/derived/models/mountain_256x256_seeds40-42_20000.synthetic_event_list_targets_h6.csv}"
OUT="${OUT:-data/derived/models/mountain_256x256_seeds40-42_20000.synthetic_event_list_targets_h6.episodes.csv}"
REPORT="${REPORT:-data/derived/models/mountain_256x256_seeds40-42_20000.synthetic_event_list_targets_h6.episodes.json}"
GROUP_FIELD="${GROUP_FIELD:-dataset_id}"
TIME_FIELD="${TIME_FIELD:-window_start_utc}"
ROWS_PER_EPISODE="${ROWS_PER_EPISODE:-24}"
TARGET_FIELD="${TARGET_FIELD:-eventlist_target_occurred}"
DROP_PARTIAL="${DROP_PARTIAL:-0}"

args=(
  -m elfquake.cli annotate-synthetic-episodes
  --input "$INPUT"
  --out "$OUT"
  --report "$REPORT"
  --group-field "$GROUP_FIELD"
  --time-field "$TIME_FIELD"
  --rows-per-episode "$ROWS_PER_EPISODE"
  --target-field "$TARGET_FIELD"
)

if [[ "$DROP_PARTIAL" != "0" ]]; then
  args+=(--drop-partial)
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 "${args[@]}"
