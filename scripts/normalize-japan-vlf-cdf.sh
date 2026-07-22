#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:?set INPUT to a downloaded raw .cdf file}"
ROOT="${ROOT:-data/derived/vlf/japan}"
STEM="${STEM:-$(basename "$INPUT" .cdf)}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli normalize-vlf-japan-cdf \
  --input "$INPUT" \
  --samples-out "$ROOT/${STEM}.samples.csv" \
  --metadata-out "$ROOT/${STEM}.metadata.json"
