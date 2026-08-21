#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/multimodal/all_italy.prospective_vlf_image_windows.csv}"
EVENTS="${EVENTS:-data/derived/ingv/events_italy_prospective.current.normalized.csv}"
OUT="${OUT:-data/derived/multimodal/all_italy.spatial_vlf_image_windows.labeled.csv}"
AS_OF="${AS_OF:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

# Catalog coverage end, not wall-clock now. `CATALOG_END` used to default to
# `AS_OF`, which asserts the event catalog covers up to this moment. The live
# 30-minute service did not fetch INGV -- only a manual run did -- so between
# refreshes the catalog fell behind while VLF anchors kept accumulating. Every
# anchor in that gap was then declared mature and labeled negative, because no
# event could be found in a catalog that had stopped. On 2026-08-21 that put 22
# real events' worth of false negatives into a held-out partition and produced
# the largest apparent VLF-era result in the project (see MISTAKES.md).
#
# `refresh-ingv-events.sh` records what the catalog was successfully asked for
# in `catalog_freshness.json`; that is the only honest coverage assertion. Fall
# back to the newest `ingested_at_utc` when the report is missing, which
# understates coverage rather than overstating it, and to `AS_OF` only when the
# events file carries no ingest stamps at all.
FRESHNESS="${FRESHNESS:-data/derived/ingv/catalog_freshness.json}"
if [[ -z "${CATALOG_END:-}" ]]; then
  CATALOG_END="$(EVENTS="$EVENTS" AS_OF="$AS_OF" FRESHNESS="$FRESHNESS" "$PYTHON_BIN" - <<'PYEOF'
import csv, json, os
from pathlib import Path

report = Path(os.environ["FRESHNESS"])
if report.is_file():
    coverage = json.loads(report.read_text(encoding="utf-8")).get("coverage_end_utc", "")
    if coverage:
        print(coverage)
        raise SystemExit(0)

rows = list(csv.DictReader(Path(os.environ["EVENTS"]).open(newline="", encoding="utf-8")))
stamps = [row["ingested_at_utc"] for row in rows if row.get("ingested_at_utc")]
print(max(stamps) if stamps else os.environ["AS_OF"])
PYEOF
)"
fi

printf 'catalog coverage end: %s (as-of %s)\n' "$CATALOG_END" "$AS_OF"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli label-spatial-multimodal-targets \
  --input "$INPUT" --events "$EVENTS" --out "$OUT" --as-of "$AS_OF" \
  --catalog-end "$CATALOG_END" --cell-degrees "${CELL_DEGREES:-1.5}" \
  --target-magnitude-min "${TARGET_MAGNITUDE_MIN:-2.5}"
