#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/multimodal/all_italy.prospective_vlf_image_windows.csv}"
EVENTS="${EVENTS:-data/derived/ingv/events_italy_prospective.current.normalized.csv}"
OUT="${OUT:-data/derived/multimodal/all_italy.spatial_vlf_image_windows.labeled.csv}"
AS_OF="${AS_OF:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

# Catalog coverage end, not wall-clock now. `CATALOG_END` used to default to
# `AS_OF`, which asserts the event catalog covers up to this moment. The live
# 30-minute service does not fetch INGV -- only `refresh-prospective-labels.sh`
# does -- so between manual refreshes the catalog falls behind while VLF anchors
# keep accumulating. Every anchor in that gap was then declared mature and
# labeled negative, because no event could be found in a catalog that had
# stopped. On 2026-08-21 that silently put 22 real events' worth of false
# negatives into the held-out partition (see MISTAKES.md).
#
# The honest coverage end is when the catalog was last fetched, which
# `ingested_at_utc` records. Derive it from the events file and fall back to
# `AS_OF` only when the column is absent.
if [[ -z "${CATALOG_END:-}" ]]; then
  CATALOG_END="$(EVENTS="$EVENTS" AS_OF="$AS_OF" "$PYTHON_BIN" - <<'PY'
import csv, os
from pathlib import Path

rows = list(csv.DictReader(Path(os.environ["EVENTS"]).open(newline="", encoding="utf-8")))
stamps = [row["ingested_at_utc"] for row in rows if row.get("ingested_at_utc")]
print(max(stamps) if stamps else os.environ["AS_OF"])
PY
)"
fi

printf 'catalog coverage end: %s (as-of %s)\n' "$CATALOG_END" "$AS_OF"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli label-spatial-multimodal-targets \
  --input "$INPUT" --events "$EVENTS" --out "$OUT" --as-of "$AS_OF" \
  --catalog-end "$CATALOG_END" --cell-degrees "${CELL_DEGREES:-1.5}" \
  --target-magnitude-min "${TARGET_MAGNITUDE_MIN:-2.5}"
