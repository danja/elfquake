#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SEEDS="${SEEDS:-40 41 42 4300 4600 4700 4800}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-20000}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/japan/events_japan_all.normalized.csv}"
ROOT="${ROOT:-data/derived/reports/avalanche-burst-seeds}"
DECAY="${DECAY:-0.99}"
GAP_STEPS="${GAP_STEPS:-120}"
THRESHOLD_QUANTILE="${THRESHOLD_QUANTILE:-0.975}"

mkdir -p "$ROOT"
event_files=()
for seed in $SEEDS; do
  prefix="data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}"
  [[ -f "${prefix}.avalanche_signal.csv" && -f "${prefix}.avalanche_activity.csv" ]] || {
    echo "error: missing avalanche inputs for seed $seed" >&2
    exit 2
  }
  output="$ROOT/seed${seed}.events.csv"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-avalanche-signal-event-list \
    --avalanche "${prefix}.avalanche_signal.csv" --activity "${prefix}.avalanche_activity.csv" \
    --out "$output" --grid-width "$WIDTH" --grid-height "$HEIGHT" \
    --burst-baseline-decay "$DECAY" --burst-threshold-quantile "$THRESHOLD_QUANTILE" \
    --burst-gap-steps "$GAP_STEPS"
  event_files+=("$seed=$output")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" - "$REAL_EVENTS" "$ROOT/summary.csv" "${event_files[@]}" <<'PY'
import csv
import sys
from pathlib import Path

from elfquake.features.signal_shape_compare import event_energy_series
from elfquake.features.signal_shape_stats import shape_stats

real = event_energy_series(series_id="real", events_csv=Path(sys.argv[1]), bin_seconds=3600)
real_stats = shape_stats(real.values, sample_seconds=real.sample_seconds)
out = Path(sys.argv[2])
rows = []
for item in sys.argv[3:]:
    seed, path_text = item.split("=", 1)
    path = Path(path_text)
    series = event_energy_series(series_id=seed, events_csv=path, bin_seconds=3600)
    stats = shape_stats(series.values, sample_seconds=series.sample_seconds)
    rows.append({
        "seed": seed,
        "event_count": str(sum(1 for _ in csv.DictReader(path.open(newline="", encoding="utf-8")))),
        "nonzero_ratio": f"{stats['nonzero_ratio']:.9f}",
        "real_nonzero_ratio": f"{real_stats['nonzero_ratio']:.9f}",
        "burst_run_count": f"{stats['burst_run_count']:.9f}",
        "real_burst_run_count": f"{real_stats['burst_run_count']:.9f}",
        "lag1_autocorrelation": f"{stats['lag1_autocorrelation']:.9f}",
        "real_lag1_autocorrelation": f"{real_stats['lag1_autocorrelation']:.9f}",
        "excess_kurtosis": f"{stats['excess_kurtosis']:.9f}",
        "real_excess_kurtosis": f"{real_stats['excess_kurtosis']:.9f}",
        "psd_slope": f"{stats['psd_slope']:.9f}",
        "real_psd_slope": f"{real_stats['psd_slope']:.9f}",
        "events_file": str(path),
    })
fields = list(rows[0])
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"summary output: {out}")
PY
