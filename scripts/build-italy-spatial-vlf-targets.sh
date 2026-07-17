#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/multimodal/all_italy.prospective_vlf_image_windows.csv}"
EVENTS="${EVENTS:-data/derived/ingv/events_italy_prospective.current.normalized.csv}"
OUT="${OUT:-data/derived/multimodal/all_italy.spatial_vlf_image_windows.labeled.csv}"
AS_OF="${AS_OF:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
CATALOG_END="${CATALOG_END:-$AS_OF}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli label-spatial-multimodal-targets \
  --input "$INPUT" --events "$EVENTS" --out "$OUT" --as-of "$AS_OF" \
  --catalog-end "$CATALOG_END" --cell-degrees "${CELL_DEGREES:-1.5}" \
  --target-magnitude-min "${TARGET_MAGNITUDE_MIN:-2.5}"
