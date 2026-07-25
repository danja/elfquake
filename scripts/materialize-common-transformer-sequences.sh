#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/common_transformer_fixture.csv}"
OUT_ROOT="${OUT_ROOT:-data/derived/models/common_transformer_fixture_sequences}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli \
  materialize-common-fixture-sequences --input "$INPUT" --out-root "$OUT_ROOT"
