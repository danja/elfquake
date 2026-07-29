#!/usr/bin/env bash
set -euo pipefail

source_stress="${SOURCE_STRESS_CSV:?set SOURCE_STRESS_CSV}"
activity="${AVALANCHE_ACTIVITY_CSV:?set AVALANCHE_ACTIVITY_CSV}"
out="${OUT_CSV:-${source_stress%.source_stress.csv}.source_stress_alignment.csv}"
local_radius="${LOCAL_RADIUS:-32}"
response_horizon="${RESPONSE_HORIZON:-120}"
baseline_decay="${BASELINE_DECAY:-0.99}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli analyze-source-stress-alignment \
  --source-stress "$source_stress" \
  --activity "$activity" \
  --out "$out" \
  --local-radius "$local_radius" \
  --response-horizon "$response_horizon" \
  --baseline-decay "$baseline_decay"
