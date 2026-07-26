#!/usr/bin/env bash
set -euo pipefail

# Compare native Japan CDF features and seismic events with one reproducible
# synthetic episode. This is a shape diagnostic, not a predictive evaluation.
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SIM_PREFIX="${SIM_PREFIX:-data/derived/sim/mountain_256x256_seed40_20000}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/japan/events_japan_all.normalized.csv}"
SYNTHETIC_EVENTS="${SYNTHETIC_EVENTS:-${SIM_PREFIX}.avalanche_events.csv}"
SERIES_OUT="${SERIES_OUT:-data/derived/reports/japan_synthetic_signal_shape_series.csv}"
PAIRS_OUT="${PAIRS_OUT:-data/derived/reports/japan_synthetic_signal_shape_pairs.csv}"

feature_args=()
while IFS= read -r path; do
  feature_args+=(--japan-vlf-feature "$path")
done < <(find "${JAPAN_FEATURE_ROOT:-data/derived/vlf/japan}" -maxdepth 1 -type f -name '*.features.csv' | sort)
[[ "${#feature_args[@]}" -gt 0 ]] || { echo "No Japan feature CSVs found" >&2; exit 2; }

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli compare-signal-shapes \
  --real-events "$REAL_EVENTS" \
  --synthetic-events "$SYNTHETIC_EVENTS" \
  "${feature_args[@]}" \
  --sim-piezo "${SIM_PREFIX}.piezo.csv" \
  --sim-avalanche "${SIM_PREFIX}.avalanche_signal.csv" \
  --event-bin-seconds "${EVENT_BIN_SECONDS:-3600}" \
  --sim-step-seconds "${SIM_STEP_SECONDS:-60}" \
  --series-out "$SERIES_OUT" \
  --pairs-out "$PAIRS_OUT"
