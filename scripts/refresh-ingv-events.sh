#!/usr/bin/env bash
# Fetch, normalize, and combine INGV events. Extracted from
# refresh-prospective-labels.sh so the live collector can run the same code.
#
# Why this exists: `elfquake-prospective.service` fires every 30 minutes and
# looked like a data collector, but it only updated VLF image features and the
# prospective window table. It never fetched events, so the target catalog only
# advanced when someone ran refresh-prospective-labels.sh by hand. On
# 2026-08-21 that left the catalog four days behind, and 22 real events were
# labeled as non-events inside a held-out partition (see MISTAKES.md).
#
# Fetching is a rolling window by default rather than the whole record: the
# combine step deduplicates by `event_id`, so short overlapping chunks merge
# cleanly and the service does not re-pull three months from INGV every half
# hour.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
AS_OF="${AS_OF:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
END="${END:-$AS_OF}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-7}"
START="${START:-$(date -u -d "$END - $LOOKBACK_DAYS days" +%Y-%m-%dT%H:%M:%SZ)}"
COMBINE_START_DATE="${COMBINE_START_DATE:-2026-06-01}"
START_DATE="${START:0:10}"
END_DATE="${END:0:10}"
FRESHNESS="${FRESHNESS:-data/derived/ingv/catalog_freshness.json}"

ITALY_CURRENT_EVENTS="${ITALY_CURRENT_EVENTS:-data/derived/ingv/events_italy_prospective.current.normalized.csv}"
CENTRAL_CURRENT_EVENTS="${CENTRAL_CURRENT_EVENTS:-data/derived/ingv/events_central_italy_prospective.current.normalized.csv}"
ALL_ITALY_EVENTS="${ALL_ITALY_EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}"
ALL_CENTRAL_EVENTS="${ALL_CENTRAL_EVENTS:-data/derived/ingv/events_central_italy_all_available.combined.normalized.csv}"

run_cli() { PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli "$@"; }

# A transient network failure must not take down the VLF capture steps that run
# after this script in the same oneshot unit. Record the failure, keep going,
# and let the freshness report carry the staleness forward where a human or a
# downstream check can see it. Never fail silently: FETCH_STATUS ends up in the
# report.
FETCH_STATUS=ok
if ! run_cli fetch-ingv-events --start "$START" --end "$END"; then
  echo "INGV fetch failed for ${START} to ${END}; continuing with existing catalog" >&2
  FETCH_STATUS=fetch_failed
fi

RAW="$(find data/raw/ingv -maxdepth 1 -name "events_italy_${START_DATE}_${END_DATE}_*.txt" | sort | tail -n 1)"
if [[ -n "$RAW" ]]; then
  run_cli normalize-ingv-events --raw "$RAW" \
    --out "data/derived/ingv/events_italy_${START_DATE}_${END_DATE}.normalized.csv"
  run_cli normalize-ingv-events --raw "$RAW" \
    --out "data/derived/ingv/events_central_italy_${START_DATE}_${END_DATE}.normalized.csv" \
    --only-region central_italy
else
  echo "No INGV raw file for ${START_DATE} to ${END_DATE}; skipping normalize" >&2
  [[ "$FETCH_STATUS" == "ok" ]] && FETCH_STATUS=no_raw_file
fi

collect_inputs() {
  local prefix="$1" min_date="$2"
  local -n out_ref="$3"
  out_ref=()
  while IFS= read -r path; do
    local name="${path##*/}"
    local chunk_start="${name#"$prefix"}"
    chunk_start="${chunk_start%%_*}"
    if [[ -n "$min_date" && "$chunk_start" < "$min_date" ]]; then continue; fi
    out_ref+=(--input "$path")
  done < <(find data/derived/ingv -maxdepth 1 -name "${prefix}*.normalized.csv" \
    ! -name "*.combined.normalized.csv" ! -name "*.current.normalized.csv" | sort)
}

collect_inputs "events_italy_" "$COMBINE_START_DATE" italy_inputs
collect_inputs "events_central_italy_" "$COMBINE_START_DATE" central_inputs
collect_inputs "events_italy_" "" all_italy_inputs
collect_inputs "events_central_italy_" "" all_central_inputs

run_cli combine-normalized-events "${italy_inputs[@]}" \
  --out "data/derived/ingv/events_italy_${COMBINE_START_DATE}_${END_DATE}.combined.normalized.csv"
run_cli combine-normalized-events "${central_inputs[@]}" \
  --out "data/derived/ingv/events_central_italy_${COMBINE_START_DATE}_${END_DATE}.combined.normalized.csv"
run_cli combine-normalized-events "${italy_inputs[@]}" --out "$ITALY_CURRENT_EVENTS"
run_cli combine-normalized-events "${central_inputs[@]}" --out "$CENTRAL_CURRENT_EVENTS"

# Full-history catalogs. Several downstream scripts read these; if only the
# COMBINE_START_DATE-scoped tables are rewritten they silently keep reading a
# catalog frozen at the last backfill (item 89).
run_cli combine-normalized-events "${all_italy_inputs[@]}" --out "$ALL_ITALY_EVENTS"
run_cli combine-normalized-events "${all_central_inputs[@]}" --out "$ALL_CENTRAL_EVENTS"

EVENTS="$ITALY_CURRENT_EVENTS" AS_OF="$AS_OF" FETCH_STATUS="$FETCH_STATUS" \
FRESHNESS="$FRESHNESS" WINDOW_START="$START" WINDOW_END="$END" \
  "$PYTHON_BIN" - <<'PY'
import csv, json, os
from datetime import datetime, timezone
from pathlib import Path

rows = list(csv.DictReader(Path(os.environ["EVENTS"]).open(newline="", encoding="utf-8")))
as_of = datetime.fromisoformat(os.environ["AS_OF"].replace("Z", "+00:00"))
ingested = [r["ingested_at_utc"] for r in rows if r.get("ingested_at_utc")]
events = [r["event_time_utc"] for r in rows if r.get("event_time_utc")]

report = {
    "schema": "elfquake.catalog_freshness.v1",
    "checked_at_utc": os.environ["AS_OF"],
    "fetch_status": os.environ["FETCH_STATUS"],
    "fetch_window_start_utc": os.environ["WINDOW_START"],
    "fetch_window_end_utc": os.environ["WINDOW_END"],
    "event_count": len(rows),
    "last_event_utc": max(events) if events else "",
    "last_ingested_utc": max(ingested) if ingested else "",
}

# Coverage end is what the catalog was successfully *asked* for, not when the
# last event happened and not the newest `ingested_at_utc`. A real quiet period
# is data; a stale catalog is not, and only an explicit coverage assertion
# separates them. `ingested_at_utc` cannot do it: combine deduplicates by
# `event_id` keeping the first occurrence, so a fetch that returns no new events
# leaves every stamp untouched and the catalog would look stale when it is
# current. On a failed fetch there is no assertion to make, so fall back to the
# newest ingest stamp, which understates coverage rather than overstating it.
if report["fetch_status"] == "ok":
    report["coverage_end_utc"] = report["fetch_window_end_utc"]
else:
    report["coverage_end_utc"] = report["last_ingested_utc"]

if report["coverage_end_utc"]:
    end = datetime.fromisoformat(report["coverage_end_utc"].replace("Z", "+00:00"))
    report["staleness_hours"] = round((as_of - end).total_seconds() / 3600.0, 3)
else:
    report["staleness_hours"] = None

out = Path(os.environ["FRESHNESS"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"fetch status: {report['fetch_status']}")
print(f"events: {report['event_count']}")
print(f"last event: {report['last_event_utc']}")
print(f"coverage end: {report['coverage_end_utc']} ({report['staleness_hours']} h stale)")
print(f"output: {out}")
PY
