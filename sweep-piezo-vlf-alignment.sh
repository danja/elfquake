#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./sweep-piezo-vlf-alignment.sh

Creates derived piezo/VLF transform variants from existing simulation CSVs,
materializes sequence manifests, and ranks each variant with the mixed-domain
real/synthetic VLF alignment diagnostic.

Environment:
  PYTHON_BIN                    default .venv/bin/python
  REAL_SEQUENCE                 default data/derived/models/cumiana_vlf_image_sequence/manifest.json
  ROOT                          default data/derived/models/piezo_vlf_alignment_sweep
  WIDTH                         default 256
  HEIGHT                        default 256
  STEPS                         default 20000
  SEEDS                         default "40 41 42"
  VARIANTS                      default "current fast_burst threshold_burst gain_burst"
  ALIGN_EPOCHS                  default 10
  MAX_SYNTHETIC_TRAIN_WINDOWS   default 6000
  CONTROL_METHODS               default "centroid random full"
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REAL_SEQUENCE="${REAL_SEQUENCE:-data/derived/models/cumiana_vlf_image_sequence/manifest.json}"
ROOT="${ROOT:-data/derived/models/piezo_vlf_alignment_sweep}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-20000}"
SEEDS="${SEEDS:-40 41 42}"
VARIANTS="${VARIANTS:-current fast_burst threshold_burst gain_burst}"
ALIGN_EPOCHS="${ALIGN_EPOCHS:-10}"
MAX_SYNTHETIC_TRAIN_WINDOWS="${MAX_SYNTHETIC_TRAIN_WINDOWS:-6000}"
CONTROL_METHODS="${CONTROL_METHODS:-centroid random full}"

mkdir -p "$ROOT"
summary="$ROOT/piezo_vlf_alignment_sweep.csv"
reports=()

variant_args() {
  case "$1" in
    current)
      printf '%s\n'
      ;;
    fast_burst)
      printf '%s\n' --highpass-decay 0.75 --envelope-decay 0.05 --envelope-mix 0.15 --burst-power 1.60 --near-threshold-weight 1.50 --near-threshold-floor 0.70 --release-mix 0.00 --gain-contrast 0.00
      ;;
    threshold_burst)
      printf '%s\n' --highpass-decay 0.85 --envelope-decay 0.10 --envelope-mix 0.20 --burst-power 1.40 --near-threshold-weight 2.00 --near-threshold-floor 0.72 --release-mix 0.02 --gain-contrast 0.00
      ;;
    gain_burst)
      printf '%s\n' --highpass-decay 0.85 --envelope-decay 0.05 --envelope-mix 0.20 --burst-power 1.50 --near-threshold-weight 1.00 --near-threshold-floor 0.75 --release-mix 0.00 --gain-contrast 0.35
      ;;
    mild_highpass)
      printf '%s\n' --highpass-decay 0.95 --envelope-decay 0.20 --envelope-mix 0.10 --burst-power 1.20 --near-threshold-weight 0.50 --near-threshold-floor 0.75 --release-mix 0.00 --gain-contrast 0.00
      ;;
    *)
      echo "error: unknown piezo/VLF alignment variant: $1" >&2
      exit 2
      ;;
  esac
}

control_args=()
for method in $CONTROL_METHODS; do
  control_args+=(--control-method "$method")
done

for variant in $VARIANTS; do
  variant_root="$ROOT/$variant"
  mkdir -p "$variant_root"
  manifest_args=()
  echo "piezo/VLF alignment variant: $variant"
  for seed in $SEEDS; do
    if [[ "$variant" == "current" ]]; then
      manifest="data/derived/models/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}_piezo_sequence/manifest.json"
      if [[ ! -f "$manifest" ]]; then
        echo "error: missing current piezo sequence manifest: $manifest" >&2
        exit 2
      fi
      manifest_args+=(--synthetic-sequence-manifest "$manifest")
    else
      input="data/derived/sim/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}.piezo.csv"
      if [[ ! -f "$input" ]]; then
        echo "error: missing simulation piezo CSV: $input" >&2
        exit 2
      fi
      transformed="$variant_root/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}.${variant}.piezo.csv"
      transform_report="$variant_root/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}.${variant}.transform.json"
      mapfile -t transform_args < <(variant_args "$variant")
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli transform-piezo-signal \
        --input "$input" \
        --out "$transformed" \
        --report "$transform_report" \
        "${transform_args[@]}"
      sequence_dir="$variant_root/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}_${variant}_piezo_sequence"
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli materialize-sequence-dataset \
        --input "$transformed" \
        --out-dir "$sequence_dir" \
        --time-field step \
        --entity-field sensor_id \
        --modality synthetic_piezo_vlf \
        --time-start-utc "2026-01-01T00:00:00Z" \
        --time-step-seconds 60
      manifest_args+=(--synthetic-sequence-manifest "$sequence_dir/manifest.json")
    fi
  done

  report="$variant_root/${variant}.mixed_domain_alignment.json"
  embeddings="$variant_root/${variant}.mixed_domain_alignment_embeddings.csv"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-mixed-domain-alignment \
    --real-sequence-manifest "$REAL_SEQUENCE" \
    "${manifest_args[@]}" \
    --out "$report" \
    --real-modality real_vlf_image \
    --synthetic-modality synthetic_piezo_vlf \
    --descriptor-profile shape \
    --lookback-steps 24 \
    --stride 1 \
    --train-fraction 0.8 \
    --mask-probability 0.15 \
    --inlier-fraction 0.25 \
    --inlier-method local \
    "${control_args[@]}" \
    --max-synthetic-train-windows "$MAX_SYNTHETIC_TRAIN_WINDOWS" \
    --coral-weight 0.1 \
    --epochs "$ALIGN_EPOCHS" \
    --learning-rate 0.0003 \
    --hidden-units 32 \
    --embedding-units 8 \
    --batch-size 64 \
    --seed 42 \
    --embeddings-out "$embeddings"
  reports+=("$variant=$report")
done

PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" - "$summary" "${reports[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for item in sys.argv[2:]:
    variant, report_path = item.split("=", 1)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    primary = report.get("primary", {})
    embedding = primary.get("embedding_comparison", {})
    reconstruction = primary.get("real_test_reconstruction", {})
    descriptor_gap = report.get("descriptor_gap", {})
    controls = report.get("control_runs", {})
    row = {
        "variant": variant,
        "status": report.get("status", ""),
        "synthetic_train_count": report.get("synthetic_train_count", ""),
        "centroid_distance": embedding.get("centroid_euclidean_distance", ""),
        "nearest_mean_distance": embedding.get("synthetic_to_real_nearest_mean_distance", ""),
        "scale_mean_absolute_delta": embedding.get("scale_mean_absolute_delta", ""),
        "real_test_masked_mse": reconstruction.get("masked_mse", ""),
        "real_test_zero_baseline_masked_mse": reconstruction.get("zero_baseline_masked_mse", ""),
        "mean_absolute_mean_delta": descriptor_gap.get("mean_absolute_mean_delta", ""),
        "mean_absolute_std_delta": descriptor_gap.get("mean_absolute_std_delta", ""),
        "centroid_control_distance": controls.get("centroid", {}).get("embedding_comparison", {}).get("centroid_euclidean_distance", ""),
        "random_control_distance": controls.get("random", {}).get("embedding_comparison", {}).get("centroid_euclidean_distance", ""),
        "full_control_distance": controls.get("full", {}).get("embedding_comparison", {}).get("centroid_euclidean_distance", ""),
        "report": report_path,
    }
    rows.append(row)

rows.sort(key=lambda row: float(row["centroid_distance"] or "inf"))
fieldnames = [
    "variant",
    "status",
    "synthetic_train_count",
    "centroid_distance",
    "nearest_mean_distance",
    "scale_mean_absolute_delta",
    "real_test_masked_mse",
    "real_test_zero_baseline_masked_mse",
    "mean_absolute_mean_delta",
    "mean_absolute_std_delta",
    "centroid_control_distance",
    "random_control_distance",
    "full_control_distance",
    "report",
]
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(
        row["variant"],
        "centroid", row["centroid_distance"],
        "mse", row["real_test_masked_mse"],
        "random", row["random_control_distance"],
    )
print(f"summary output: {out}")
PY
