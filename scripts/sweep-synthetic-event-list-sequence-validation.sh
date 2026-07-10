#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data/derived/models/synthetic_event_list_sequence_validation_sweep}" \
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.2}" \
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}" \
LOOKBACK_ROWS_LIST="${LOOKBACK_ROWS_LIST:-12}" \
DROPOUTS="${DROPOUTS:-0.1 0.2}" \
SEEDS="${SEEDS:-7 42 99}" \
  ./scripts/sweep-synthetic-event-list-sequence-head.sh
