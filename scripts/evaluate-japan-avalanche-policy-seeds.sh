#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SEEDS="${SEEDS:-40 41 42 4300 4600 4700 4800}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-20000}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/japan/events_japan_all.normalized.csv}"
ROOT="${ROOT:-data/derived/reports/japan-avalanche-policy-seeds}"
QUANTILE="${QUANTILE:-0.975}"
LOCAL_MAX_WINDOW="${LOCAL_MAX_WINDOW:-120}"
MAX_EVENTS="${MAX_EVENTS:-25}"

mkdir -p "$ROOT"
event_files=()
for seed in $SEEDS; do
  prefix="data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}"
  for suffix in avalanche_signal.csv avalanche_activity.csv; do
    [[ -f "${prefix}.${suffix}" ]] || { echo "error: missing ${prefix}.${suffix}" >&2; exit 2; }
  done
  output="$ROOT/${seed}.avalanche_events.csv"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-avalanche-signal-event-list \
    --avalanche "${prefix}.avalanche_signal.csv" \
    --activity "${prefix}.avalanche_activity.csv" \
    --out "$output" \
    --grid-width "$WIDTH" --grid-height "$HEIGHT" \
    --min-signal-quantile "$QUANTILE" \
    --local-max-window "$LOCAL_MAX_WINDOW" \
    --max-events "$MAX_EVENTS"
  event_files+=("$seed=$output")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" - "$REAL_EVENTS" "$ROOT/summary.csv" "$QUANTILE" "$LOCAL_MAX_WINDOW" "$MAX_EVENTS" "${event_files[@]}" <<'PY'
import csv
import sys
from pathlib import Path

from elfquake.features.signal_shape_compare import event_energy_series
from elfquake.features.signal_shape_stats import shape_stats

real_path = Path(sys.argv[1])
out = Path(sys.argv[2])
quantile, window, max_events = sys.argv[3:6]
real = event_energy_series(series_id="real", events_csv=real_path, bin_seconds=3600)
real_stats = shape_stats(real.values, sample_seconds=real.sample_seconds)
rows = []
for item in sys.argv[6:]:
    seed, path_text = item.split("=", 1)
    path = Path(path_text)
    series = event_energy_series(series_id=seed, events_csv=path, bin_seconds=3600)
    stats = shape_stats(series.values, sample_seconds=series.sample_seconds)
    rows.append({
        "seed": seed,
        "event_count": sum(1 for _ in csv.DictReader(path.open(newline="", encoding="utf-8"))),
        "real_event_count": sum(1 for _ in csv.DictReader(real_path.open(newline="", encoding="utf-8"))),
        "synthetic_nonzero_ratio": f"{stats['nonzero_ratio']:.9f}",
        "real_nonzero_ratio": f"{real_stats['nonzero_ratio']:.9f}",
        "nonzero_delta": f"{stats['nonzero_ratio'] - real_stats['nonzero_ratio']:.9f}",
        "synthetic_burst_run_count": f"{stats['burst_run_count']:.9f}",
        "real_burst_run_count": f"{real_stats['burst_run_count']:.9f}",
        "synthetic_lag1_autocorrelation": f"{stats['lag1_autocorrelation']:.9f}",
        "real_lag1_autocorrelation": f"{real_stats['lag1_autocorrelation']:.9f}",
        "synthetic_excess_kurtosis": f"{stats['excess_kurtosis']:.9f}",
        "real_excess_kurtosis": f"{real_stats['excess_kurtosis']:.9f}",
        "synthetic_psd_slope": f"{stats['psd_slope']:.9f}",
        "real_psd_slope": f"{real_stats['psd_slope']:.9f}",
        "events_file": str(path),
    })
fields = ["seed", "event_count", "real_event_count", "synthetic_nonzero_ratio", "real_nonzero_ratio", "nonzero_delta", "synthetic_burst_run_count", "real_burst_run_count", "synthetic_lag1_autocorrelation", "real_lag1_autocorrelation", "synthetic_excess_kurtosis", "real_excess_kurtosis", "synthetic_psd_slope", "real_psd_slope", "events_file"]
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"summary output: {out}")
PY
