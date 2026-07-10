#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data/derived/models/synthetic_event_list_sequence_early_stop_sweep}" \
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.2}" \
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}" \
CALIBRATION_SOURCE="${CALIBRATION_SOURCE:-train}" \
LOOKBACK_ROWS_LIST="${LOOKBACK_ROWS_LIST:-12}" \
DROPOUTS="${DROPOUTS:-0.1 0.2}" \
SEEDS="${SEEDS:-7 42 99}" \
EPOCHS="${EPOCHS:-120}" \
  ./scripts/sweep-synthetic-event-list-sequence-head.sh
