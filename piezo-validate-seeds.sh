#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./piezo-validate-seeds.sh

Validates the current tuned piezo defaults across multiple seeds.

Environment:
  SEEDS            default "40 41 42"
  WIDTH            default 256
  HEIGHT           default 256
  STEPS            default 10000
  RUN_SIM          default missing; use 1 to force reruns, 0 to reuse existing CSVs
  REAL_IMAGE_ROOT  default data/raw/vlf/cumiana/captures
  SUMMARY_OUT      default data/derived/sim/piezo_seed_validation_summary.csv
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

seeds="${SEEDS:-40 41 42}"
width="${WIDTH:-256}"
height="${HEIGHT:-256}"
steps="${STEPS:-10000}"
run_sim="${RUN_SIM:-missing}"
real_image_root="${REAL_IMAGE_ROOT:-data/raw/vlf/cumiana/captures}"
summary_out="${SUMMARY_OUT:-data/derived/sim/piezo_seed_validation_summary.csv}"

scan_files=()

for seed in $seeds; do
  prefix="data/derived/sim/mountain_${width}x${height}_seed${seed}_${steps}"
  piezo_csv="${prefix}.piezo.csv"
  echo "piezo seed validation: seed $seed"
  if [[ "$run_sim" == "1" || ( "$run_sim" == "missing" && ! -f "$piezo_csv" ) ]]; then
    WIDTH="$width" HEIGHT="$height" STEPS="$steps" SEED="$seed" RUN_HEATMAPS=0 ./sim.sh
  elif [[ "$run_sim" == "0" || "$run_sim" == "missing" ]]; then
    if [[ ! -f "$piezo_csv" ]]; then
      echo "error: missing piezo CSV for seed $seed: $piezo_csv" >&2
      exit 2
    fi
    echo "simulation reused: $piezo_csv"
  else
    echo "error: RUN_SIM must be 1, 0, or missing" >&2
    exit 2
  fi

  VLF_IMAGE_ROOT="$real_image_root" ./piezo-sensor-scan.sh \
    "$piezo_csv" \
    "${prefix}.piezo_sensor_scan.csv"
  scan_files+=("${prefix}.piezo_sensor_scan.csv")
done

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - "$summary_out" "${scan_files[@]}" <<'PY'
import csv
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
fieldnames = [
    "seed",
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
    match = re.search(r"_seed(\d+)_", path.name)
    seed = match.group(1) if match else ""
    with path.open(newline="", encoding="utf-8") as handle:
        best = next(csv.DictReader(handle))
    row = {"seed": seed, "scan_file": str(path)}
    row.update({field: best.get(field, "") for field in fieldnames if field not in row})
    rows.append(row)

rows.sort(key=lambda row: int(row["seed"]) if row["seed"] else -1)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(
        "seed", row["seed"],
        "sensor", row["sensor_id"],
        "score", row["shape_score"],
        "lag", row["lag1_autocorrelation"],
        "psd", row["psd_slope"],
        "burst_rate", row["burst_run_rate"],
    )
print(f"summary output: {out}")
PY
