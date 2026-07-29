#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ROOT:-data/derived/models/piezo_vlf_alignment_seed_holdout}"
REAL_SEQUENCE="${REAL_SEQUENCE:-data/derived/models/cumiana_vlf_image_sequence/manifest.json}"
SEEDS="${SEEDS:-40 41 42}"
VARIANTS="${VARIANTS:-current gain_burst fast_burst}"
ALIGN_SEEDS="${ALIGN_SEEDS:-42 99}"
mkdir -p "$ROOT"

for variant in $VARIANTS; do
  for seed in $SEEDS; do
    if [[ "$variant" == "current" ]]; then
      manifest="data/derived/models/mountain_256x256_seed${seed}_20000_piezo_sequence/manifest.json"
    else
      manifest="data/derived/models/piezo_vlf_alignment_sweep/${variant}/mountain_256x256_seed${seed}_20000_${variant}_piezo_sequence/manifest.json"
    fi
    [[ -f "$manifest" ]] || { echo "error: missing manifest: $manifest" >&2; exit 2; }
    for align_seed in $ALIGN_SEEDS; do
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-mixed-domain-alignment \
      --real-sequence-manifest "$REAL_SEQUENCE" \
      --synthetic-sequence-manifest "$manifest" \
      --out "$ROOT/${variant}_seed${seed}_modelseed${align_seed}.json" \
      --real-modality real_vlf_image \
      --synthetic-modality synthetic_piezo_vlf \
      --descriptor-profile shape \
      --lookback-steps 24 --stride 1 --train-fraction 0.8 \
      --mask-probability 0.15 --inlier-fraction 0.25 --inlier-method local \
      --control-method centroid --control-method random --control-method full \
      --max-synthetic-train-windows 6000 --coral-weight 0.1 \
      --epochs "${ALIGN_EPOCHS:-10}" --learning-rate 0.0003 \
      --hidden-units 32 --embedding-units 8 --batch-size 64 --seed "$align_seed" \
      --embeddings-out "$ROOT/${variant}_seed${seed}_modelseed${align_seed}.embeddings.csv"
    done
  done
done

PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" - "$ROOT" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    comparison = report["primary"]["embedding_comparison"]
    rows.append({
        "run": path.stem,
        "variant": path.stem.split("_seed", 1)[0],
        "simulation_seed": path.stem.split("_seed", 1)[1].split("_modelseed", 1)[0],
        "model_seed": path.stem.split("_modelseed", 1)[1],
        "centroid_distance": comparison.get("centroid_euclidean_distance", ""),
        "nearest_mean_distance": comparison.get("synthetic_to_real_nearest_mean_distance", ""),
        "scale_mean_absolute_delta": comparison.get("scale_mean_absolute_delta", ""),
        "status": report.get("status", ""),
    })
with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
groups = defaultdict(list)
for row in rows:
    groups[row["variant"]].append(row)
for variant, values in sorted(groups.items()):
    print(variant)
    for field in ("centroid_distance", "nearest_mean_distance", "scale_mean_absolute_delta"):
        numbers = [float(value[field]) for value in values]
        print(f"  {field}: mean={sum(numbers)/len(numbers):.6f} min={min(numbers):.6f} max={max(numbers):.6f}")
print(f"summary output: {root / 'summary.csv'}")
PY
