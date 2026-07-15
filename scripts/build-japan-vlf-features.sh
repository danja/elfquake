#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
IMAGE_ROOT="${IMAGE_ROOT:-data/raw/vlf/japan/captures}"
OUT="${OUT:-data/derived/japan/japan_vlf.image_features.csv}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli extract-vlf-image-features \
  --image-root "$IMAGE_ROOT" --out "$OUT"
