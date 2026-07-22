#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:?Set INPUT to a raw ISEE VLF CDF file}"
OUTPUT="${OUTPUT:-data/derived/vlf/japan/$(basename "$INPUT" .cdf).features.csv}"
BANDS="${BANDS:-8}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli extract-vlf-japan-cdf-features \
  --input "$INPUT" --out "$OUTPUT" --bands "$BANDS"
