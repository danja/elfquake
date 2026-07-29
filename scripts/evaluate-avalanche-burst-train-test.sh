#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
TRAIN_SEEDS="${TRAIN_SEEDS:-40 41 42 4300}"
TEST_SEEDS="${TEST_SEEDS:-4600 4700 4800}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-20000}"
ROOT="${ROOT:-data/derived/reports/avalanche-burst-train-test}"
DECAY="${DECAY:-0.99}"
THRESHOLD_QUANTILE="${THRESHOLD_QUANTILE:-0.975}"
GAP_STEPS="${GAP_STEPS:-120}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/japan/events_japan_all.normalized.csv}"

mkdir -p "$ROOT"
train_inputs=()
for seed in $TRAIN_SEEDS; do
  train_inputs+=("data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}.avalanche_signal.csv")
done
threshold=$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" - "$DECAY" "$THRESHOLD_QUANTILE" "${train_inputs[@]}" <<'PY'
import sys
from elfquake.sim.avalanche_bursts import quantile_threshold
from elfquake.sim.synthetic_events import read_avalanche_burst_scores

decay = float(sys.argv[1])
fraction = float(sys.argv[2])
scores = []
for path in sys.argv[3:]:
    scores.extend(read_avalanche_burst_scores(
        avalanche_csv=__import__('pathlib').Path(path), baseline_decay=decay, relative_baseline=True
    ))
print(f"{quantile_threshold(scores, fraction):.9f}")
PY
)
echo "train-only relative burst threshold: $threshold"

seed_rows=()
for split in train test; do
  seeds="${TRAIN_SEEDS:-}"; [[ "$split" == test ]] && seeds="$TEST_SEEDS"
  for seed in $seeds; do
    prefix="data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}"
    [[ -f "${prefix}.avalanche_signal.csv" && -f "${prefix}.avalanche_activity.csv" ]] || {
      echo "error: missing avalanche inputs for seed $seed" >&2; exit 2;
    }
    output="$ROOT/${split}_seed${seed}.events.csv"
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-avalanche-signal-event-list \
      --avalanche "${prefix}.avalanche_signal.csv" --activity "${prefix}.avalanche_activity.csv" \
      --out "$output" --grid-width "$WIDTH" --grid-height "$HEIGHT" \
      --burst-baseline-decay "$DECAY" --burst-relative-baseline \
      --burst-threshold "$threshold" --burst-gap-steps "$GAP_STEPS"
    seed_rows+=("$split:$seed=$output")
  done
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" - "$REAL_EVENTS" "$ROOT/summary.csv" "$threshold" "${seed_rows[@]}" <<'PY'
import csv
import sys
from pathlib import Path
from elfquake.features.signal_shape_compare import event_energy_series
from elfquake.features.signal_shape_stats import shape_stats

real = event_energy_series(series_id="real", events_csv=Path(sys.argv[1]), bin_seconds=3600)
real_stats = shape_stats(real.values, sample_seconds=real.sample_seconds)
out = Path(sys.argv[2])
rows = []
threshold = sys.argv[3]
for item in sys.argv[4:]:
    split_seed, path_text = item.split("=", 1)
    split, seed = split_seed.split(":", 1)
    path = Path(path_text)
    series = event_energy_series(series_id=seed, events_csv=path, bin_seconds=3600)
    stats = shape_stats(series.values, sample_seconds=series.sample_seconds)
    rows.append({
        "split": split, "seed": seed, "threshold": threshold,
        "event_count": str(sum(1 for _ in csv.DictReader(path.open(newline="", encoding="utf-8")))),
        "nonzero_ratio": f"{stats['nonzero_ratio']:.9f}",
        "real_nonzero_ratio": f"{real_stats['nonzero_ratio']:.9f}",
        "burst_run_count": f"{stats['burst_run_count']:.9f}",
        "real_burst_run_count": f"{real_stats['burst_run_count']:.9f}",
        "lag1_autocorrelation": f"{stats['lag1_autocorrelation']:.9f}",
        "excess_kurtosis": f"{stats['excess_kurtosis']:.9f}",
        "psd_slope": f"{stats['psd_slope']:.9f}",
        "events_file": str(path),
    })
fields = list(rows[0])
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
print(f"summary output: {out}")
PY
