#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/common_transformer_fixture.csv}"
OUT="${OUT:-data/derived/models/common_transformer_fixture.alignment_audit.json}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli audit-common-window-alignment \
  --input "$INPUT" \
  --out "$OUT"
