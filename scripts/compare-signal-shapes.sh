#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/compare-signal-shapes.sh

Writes time-domain and frequency-domain shape metrics for real and synthetic signals.

Environment:
  WIDTH             default 256
  HEIGHT            default 256
  STEPS             default 10000
  SEED              default 42
  REAL_EVENTS       normalized INGV-like CSV; default newest data/derived/ingv/*.normalized.csv
  SYNTHETIC_EVENTS  synthetic seismic event CSV; default current avalanche_events.csv
  REAL_VLF_ROOT     default data/raw/vlf/cumiana/captures
  SIM_PIEZO         default current piezo.csv
  SIM_AVALANCHE     default current avalanche_signal.csv, falling back to piezo_avalanche.csv
  EVENT_BIN_SECONDS default 3600
  SIM_STEP_SECONDS  default 60
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

width="${WIDTH:-256}"
height="${HEIGHT:-256}"
steps="${STEPS:-10000}"
seed="${SEED:-42}"
prefix="data/derived/sim/mountain_${width}x${height}_seed${seed}_${steps}"

real_events="${REAL_EVENTS:-}"
if [[ -z "$real_events" ]]; then
  real_events="$(find data/derived/ingv -type f -name '*.normalized.csv' -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1s/^[^ ]* //p')"
fi

synthetic_events="${SYNTHETIC_EVENTS:-${prefix}.avalanche_events.csv}"
real_vlf_root="${REAL_VLF_ROOT:-data/raw/vlf/cumiana/captures}"
sim_piezo="${SIM_PIEZO:-${prefix}.piezo.csv}"
sim_avalanche="${SIM_AVALANCHE:-${prefix}.avalanche_signal.csv}"
if [[ ! -f "$sim_avalanche" && -f "${prefix}.piezo_avalanche.csv" ]]; then
  sim_avalanche="${prefix}.piezo_avalanche.csv"
fi
event_bin_seconds="${EVENT_BIN_SECONDS:-3600}"
sim_step_seconds="${SIM_STEP_SECONDS:-60}"
series_out="${SERIES_OUT:-${prefix}.signal_shape_series.csv}"
pairs_out="${PAIRS_OUT:-${prefix}.signal_shape_pairs.csv}"

args=(
  compare-signal-shapes
  --event-bin-seconds "$event_bin_seconds"
  --sim-step-seconds "$sim_step_seconds"
  --series-out "$series_out"
  --pairs-out "$pairs_out"
)

if [[ -n "$real_events" && -f "$real_events" ]]; then
  args+=(--real-events "$real_events")
else
  echo "warning: no normalized real INGV event CSV found; set REAL_EVENTS=..." >&2
fi

if [[ -f "$synthetic_events" ]]; then
  args+=(--synthetic-events "$synthetic_events")
else
  echo "warning: synthetic event CSV not found: $synthetic_events" >&2
fi

if [[ -d "$real_vlf_root" ]]; then
  args+=(--real-vlf-image-root "$real_vlf_root")
else
  echo "warning: real VLF image root not found: $real_vlf_root" >&2
fi

if [[ -f "$sim_piezo" ]]; then
  args+=(--sim-piezo "$sim_piezo")
else
  echo "warning: simulated piezo CSV not found: $sim_piezo" >&2
fi

if [[ -f "$sim_avalanche" ]]; then
  args+=(--sim-avalanche "$sim_avalanche")
else
  echo "warning: simulated avalanche signal CSV not found: $sim_avalanche" >&2
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli "${args[@]}"
