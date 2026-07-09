#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

synthetic_manifests=()
for seed in 40 41 42; do
  synthetic_manifests+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_avalanche_sequence/manifest.json")
  synthetic_manifests+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_piezo_sequence/manifest.json")
  synthetic_manifests+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_summary_sequence/manifest.json")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli estimate-model-scale \
  --input data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv \
  "${synthetic_manifests[@]}" \
  --out data/derived/models/model_scale_synthetic_20000.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli estimate-model-scale \
  --input data/derived/models/sequence_full_balanced/mountain_256x256_seeds40-42_20000.regime_balanced_split.csv \
  "${synthetic_manifests[@]}" \
  --out data/derived/models/model_scale_synthetic_balanced_20000.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli estimate-model-scale \
  --input data/derived/models/all_italy.real_vlf_aligned_windows.csv \
  --sequence-manifest data/derived/models/cumiana_vlf_image_sequence/manifest.json \
  --out data/derived/models/model_scale_all_italy_real_vlf.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli estimate-model-scale \
  --input data/derived/models/central_italy.real_vlf_aligned_windows.csv \
  --sequence-manifest data/derived/models/cumiana_vlf_image_sequence/manifest.json \
  --out data/derived/models/model_scale_central_italy_real_vlf.json
