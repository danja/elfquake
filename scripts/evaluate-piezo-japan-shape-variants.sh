#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/sim/mountain_256x256_seed40_20000.piezo.csv}"
EVENTS="${EVENTS:-data/derived/japan/events_japan_all.normalized.csv}"
SYNTHETIC_EVENTS="${SYNTHETIC_EVENTS:-data/derived/sim/mountain_256x256_seed40_20000.avalanche_events.csv}"
FEATURE_ROOT="${FEATURE_ROOT:-data/derived/vlf/japan}"
ROOT="${ROOT:-data/derived/reports/piezo-japan-shape-variants}"

if [[ ! -f "$INPUT" || ! -f "$EVENTS" || ! -f "$SYNTHETIC_EVENTS" ]]; then
  echo "error: required simulation or Japan seismic input is missing" >&2
  exit 2
fi
mapfile -t FEATURES < <(find "$FEATURE_ROOT" -name '*.features.csv' -type f | sort)
if [[ "${#FEATURES[@]}" -eq 0 ]]; then
  echo "error: no Japan VLF feature captures found under $FEATURE_ROOT" >&2
  exit 2
fi

mkdir -p "$ROOT"
reports=()
for variant in baseline slow_weak slow_strong slow_long; do
  case "$variant" in
    baseline) args=(--slow-envelope-mix 0.0) ;;
    slow_weak) args=(--slow-envelope-decay 0.995 --slow-envelope-mix 0.4) ;;
    slow_strong) args=(--slow-envelope-decay 0.995 --slow-envelope-mix 0.8) ;;
    slow_long) args=(--slow-envelope-decay 0.999 --slow-envelope-mix 0.6) ;;
  esac
  output="$ROOT/${variant}.piezo.csv"
  transform_report="$ROOT/${variant}.transform.json"
  series="$ROOT/${variant}.series.csv"
  pairs="$ROOT/${variant}.pairs.csv"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli transform-piezo-signal \
    --input "$INPUT" --out "$output" --report "$transform_report" "${args[@]}"
  feature_args=()
  for feature in "${FEATURES[@]}"; do
    feature_args+=(--japan-vlf-feature "$feature")
  done
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli compare-signal-shapes \
    --real-events "$EVENTS" --synthetic-events "$SYNTHETIC_EVENTS" \
    "${feature_args[@]}" --sim-piezo "$output" \
    --series-out "$series" --pairs-out "$pairs"
  reports+=("$variant=$series")
done

PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" - "$ROOT/summary.csv" "${reports[@]}" <<'PY'
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for item in sys.argv[2:]:
    variant, path = item.split("=", 1)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        records = {row["series_id"]: row for row in csv.DictReader(handle)}
    japan = records["japan_vlf_cdf_log_power"]
    piezo = records["synthetic_piezo_vlf_signal"]
    rows.append({
        "variant": variant,
        "japan_psd_slope": japan["psd_slope"],
        "piezo_psd_slope": piezo["psd_slope"],
        "japan_psd_low_band_ratio": japan["psd_low_band_ratio"],
        "piezo_psd_low_band_ratio": piezo["psd_low_band_ratio"],
        "japan_lag1_autocorrelation": japan["lag1_autocorrelation"],
        "piezo_lag1_autocorrelation": piezo["lag1_autocorrelation"],
        "piezo_excess_kurtosis": piezo["excess_kurtosis"],
        "series": path,
    })
fields = list(rows[0])
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(row["variant"], "slope", row["piezo_psd_slope"], "low", row["piezo_psd_low_band_ratio"])
print(f"summary output: {out}")
PY
