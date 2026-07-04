#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./piezo-tune-grid.sh

Runs a small piezo-parameter grid without overwriting default simulation files.

Environment:
  WIDTH            default 256
  HEIGHT           default 256
  STEPS            default 10000
  SEED             default 42
  VARIANTS         default "threshold40_local16 threshold40_local8 threshold35_local16"
  RUN_SIM          default 1, set 0 to reuse existing variant CSVs
  REAL_IMAGE_ROOT  default data/raw/vlf/cumiana/captures
  SUMMARY_OUT      default data/derived/sim/piezo_tuning_summary.csv
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
variants="${VARIANTS:-threshold40_local16 threshold40_local8 threshold35_local16}"
run_sim="${RUN_SIM:-1}"
real_image_root="${REAL_IMAGE_ROOT:-data/raw/vlf/cumiana/captures}"
summary_out="${SUMMARY_OUT:-data/derived/sim/piezo_tuning_summary.csv}"

source_count="${SOURCE_COUNT:-$(( width * height / 64 ))}"
if [[ "$source_count" -lt 16 ]]; then
  source_count=16
fi
target_fill_limit="${TARGET_FILL_LIMIT:-$(( width * height / 16 ))}"
if [[ "$target_fill_limit" -lt 1 ]]; then
  target_fill_limit=1
fi
slope_threshold="${SLOPE_THRESHOLD:-$(( width / 16 ))}"
if [[ "$slope_threshold" -lt 4 ]]; then
  slope_threshold=4
fi

mkdir -p "$(dirname "$summary_out")"
scan_files=()

for variant in $variants; do
  case "$variant" in
    default)
      release_charge_threshold=0
      attenuation_radius=0
      max_distance_radius=0
      release_ratio=0.15
      critical_release_ratio=0.05
      charge_decay=0.995
      ;;
    threshold40)
      release_charge_threshold=40
      attenuation_radius=0
      max_distance_radius=0
      release_ratio=0.15
      critical_release_ratio=0.05
      charge_decay=0.995
      ;;
    threshold40_local16)
      release_charge_threshold=40
      attenuation_radius=16
      max_distance_radius=48
      release_ratio=0.25
      critical_release_ratio=0.10
      charge_decay=0.995
      ;;
    threshold40_local8)
      release_charge_threshold=40
      attenuation_radius=8
      max_distance_radius=24
      release_ratio=0.30
      critical_release_ratio=0.12
      charge_decay=0.995
      ;;
    threshold35_local16)
      release_charge_threshold=35
      attenuation_radius=16
      max_distance_radius=48
      release_ratio=0.25
      critical_release_ratio=0.10
      charge_decay=0.995
      ;;
    threshold35_local8)
      release_charge_threshold=35
      attenuation_radius=8
      max_distance_radius=24
      release_ratio=0.30
      critical_release_ratio=0.12
      charge_decay=0.995
      ;;
    *)
      echo "error: unknown piezo tuning variant: $variant" >&2
      exit 2
      ;;
  esac

  prefix="data/derived/sim/mountain_${width}x${height}_seed${seed}_${steps}_piezo_${variant}"
  echo "piezo tuning variant: $variant"
  if [[ "$run_sim" != "0" ]]; then
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m elfquake.cli run-sandpile-sim \
      --mountain-mode \
      --width "$width" --height "$height" --steps "$steps" \
      --threshold "$slope_threshold" \
      --deposition-mode sources \
      --source-count "$source_count" --sensor-count 16 \
      --deposition-probability 0.7 --seed "$seed" \
      --target-fill-limit "$target_fill_limit" \
      --bottom-layer-removal-interval 100 \
      --summary-out "${prefix}.summary.csv" \
      --sensors-out "${prefix}.sensors.csv" \
      --piezo-out "${prefix}.piezo.csv" \
      --avalanche-signal-out "${prefix}.avalanche_signal.csv" \
      --avalanche-activity-out "${prefix}.avalanche_activity.csv" \
      --piezo-sensor-count 16 \
      --piezo-cluster-count 8 \
      --piezo-activation-ratio 0.75 \
      --piezo-susceptibility-base 0.15 \
      --piezo-susceptibility-variation 0.85 \
      --piezo-attenuation-radius "$attenuation_radius" \
      --piezo-max-distance-radius "$max_distance_radius" \
      --piezo-charge-decay "$charge_decay" \
      --piezo-charge-coupling 1.0 \
      --piezo-release-charge-threshold "$release_charge_threshold" \
      --piezo-release-ratio "$release_ratio" \
      --piezo-critical-release-ratio "$critical_release_ratio" \
      --piezo-saturation 1000 \
      --progress-interval 1000
  fi

  VLF_IMAGE_ROOT="$real_image_root" ./piezo-sensor-scan.sh \
    "${prefix}.piezo.csv" \
    "${prefix}.piezo_sensor_scan.csv"
  scan_files+=("${prefix}.piezo_sensor_scan.csv")
done

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - "$summary_out" "${scan_files[@]}" <<'PY'
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])
fieldnames = [
    "variant",
    "sensor_id",
    "shape_score",
    "lag1_autocorrelation",
    "delta_lag1_autocorrelation",
    "psd_slope",
    "delta_psd_slope",
    "burst_run_rate",
    "delta_burst_run_rate",
    "coefficient_variation",
    "delta_coefficient_variation",
    "nonzero_ratio",
    "delta_nonzero_ratio",
    "scan_file",
]
rows = []
for scan_file in sys.argv[2:]:
    path = Path(scan_file)
    with path.open(newline="", encoding="utf-8") as handle:
        best = next(csv.DictReader(handle))
    stem = path.name.split("_piezo_", 1)[-1].replace(".piezo_sensor_scan.csv", "")
    row = {"variant": stem, "scan_file": str(path)}
    row.update({field: best.get(field, "") for field in fieldnames if field not in row})
    rows.append(row)

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(
        row["variant"],
        "sensor", row["sensor_id"],
        "score", row["shape_score"],
        "lag", row["lag1_autocorrelation"],
        "psd", row["psd_slope"],
        "burst_rate", row["burst_run_rate"],
    )
print(f"summary output: {out}")
PY
