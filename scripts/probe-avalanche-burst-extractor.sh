#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PREFIX="${PREFIX:-data/derived/sim/mountain_256x256_seed40_20000}"
ROOT="${ROOT:-data/derived/reports/avalanche-burst-extractor}"
mkdir -p "$ROOT"

variants=("decay995_gap30:0.995:30" "decay995_gap120:0.995:120" "decay99_gap30:0.99:30" "decay99_gap120:0.99:120")
reports=()
for item in "${variants[@]}"; do
  IFS=: read -r name decay gap <<< "$item"
  output="$ROOT/${name}.events.csv"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-avalanche-signal-event-list \
    --avalanche "${PREFIX}.avalanche_signal.csv" \
    --activity "${PREFIX}.avalanche_activity.csv" \
    --out "$output" --grid-width 256 --grid-height 256 \
    --burst-baseline-decay "$decay" --burst-threshold-quantile 0.975 \
    --burst-gap-steps "$gap" --max-events 0
  reports+=("$name=$output")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" - "$ROOT/summary.csv" "${reports[@]}" <<'PY'
import csv
import sys
from pathlib import Path

from elfquake.features.signal_shape_compare import event_energy_series
from elfquake.features.signal_shape_stats import shape_stats

out = Path(sys.argv[1])
rows = []
for item in sys.argv[2:]:
    name, path_text = item.split("=", 1)
    path = Path(path_text)
    series = event_energy_series(series_id=name, events_csv=path, bin_seconds=3600)
    stats = shape_stats(series.values, sample_seconds=series.sample_seconds)
    rows.append({
        "variant": name,
        "event_count": str(sum(1 for _ in csv.DictReader(path.open(newline="", encoding="utf-8")))),
        "nonzero_ratio": f"{stats['nonzero_ratio']:.9f}",
        "burst_run_count": f"{stats['burst_run_count']:.9f}",
        "lag1_autocorrelation": f"{stats['lag1_autocorrelation']:.9f}",
        "excess_kurtosis": f"{stats['excess_kurtosis']:.9f}",
        "psd_slope": f"{stats['psd_slope']:.9f}",
        "events_file": str(path),
    })
fields = list(rows[0])
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"summary output: {out}")
PY
