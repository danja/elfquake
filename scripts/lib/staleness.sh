# Freshness guard for derived inputs (item 89 / item 95(d)).
#
# The item-89 defect class is a script that reads one input which a refresh
# rebuilds and another which nothing rebuilds. The run looks current, the
# report carries today's date, and part of the answer is weeks old. It has
# now happened three times in this repo: the `all_available` catalogs frozen
# for a month, `prepare-italy-spatial-model-inputs.sh` reusing Aug-4 targets,
# and the live collector never fetching INGV at all (item 104).
#
# Source this file and call `require_fresh_inputs` before doing any work. It
# compares modification times only; it does not rebuild anything, because the
# inputs it guards are model artifacts that cost far more than the report
# consuming them and the decision to regenerate belongs to the caller.
#
# Usage:
#   . "$(dirname "$0")/lib/staleness.sh"
#   require_fresh_inputs "$EVENTS" "$ANOMALY" "$VLF_WINDOWS"
#
# The first argument is the reference: the input the refresh pipeline does
# keep current, normally the INGV event catalog. Every later argument is
# checked against it. Default behaviour is a loud warning on stderr, so an
# audit run still produces its report; set STALE_INPUTS=fail to turn that
# into exit status 3 for unattended or CI use.

require_fresh_inputs() {
  local reference="$1"
  shift
  local mode="${STALE_INPUTS:-warn}"

  if [ ! -e "$reference" ]; then
    echo "staleness guard: reference '$reference' does not exist; skipping check" >&2
    return 0
  fi

  local stale=()
  local input
  for input in "$@"; do
    # A missing input is a different failure and the command below will report
    # it in its own terms. Only an existing, older file is staleness.
    if [ -e "$input" ] && [ "$input" -ot "$reference" ]; then
      stale+=("$input")
    fi
  done

  if [ "${#stale[@]}" -eq 0 ]; then
    return 0
  fi

  # Cap the listing. A caller that passes a glob can match dozens of orphaned
  # artifacts from datasets it does not read, and a 40-line warning trains the
  # operator to skip warnings. The count above the list is the real signal.
  local limit="${STALE_INPUTS_MAX_LISTED:-8}"
  {
    echo "staleness guard: ${#stale[@]} input(s) older than the reference."
    echo "  reference $(_staleness_stamp "$reference")  $reference"
    local shown=0
    for input in "${stale[@]}"; do
      if [ "$shown" -ge "$limit" ]; then
        echo "  ... and $(( ${#stale[@]} - shown )) more"
        break
      fi
      echo "  stale     $(_staleness_stamp "$input")  $input"
      shown=$((shown + 1))
    done
    echo "  Nothing rebuilds these on the refresh path. Any score they produce"
    echo "  describes the record as of the older timestamp, not today's."
  } >&2

  if [ "$mode" = "fail" ]; then
    return 3
  fi
  return 0
}

_staleness_stamp() {
  date -u -r "$1" +%Y-%m-%dT%H:%MZ 2>/dev/null || echo "unknown-time"
}

# Coverage end of the INGV catalog, as recorded by `refresh-ingv-events.sh`.
#
# Prints `coverage_end_utc` from the freshness report, or the fallback when no
# report exists. Never derive this from wall-clock time and never from
# max(ingested_at_utc): `combine-normalized-events` deduplicates by event_id
# and keeps the first occurrence, so a fetch that finds no new events leaves
# every ingest stamp untouched and a current catalog looks stale. A quiet
# period is data, not absence of coverage.
#
# Usage: CATALOG_END="$(catalog_coverage_end "$FRESHNESS" "$END")"
catalog_coverage_end() {
  local report="${1:-data/derived/ingv/catalog_freshness.json}"
  local fallback="$2"
  FRESHNESS="$report" FALLBACK="$fallback" "${PYTHON_BIN:-.venv/bin/python}" - <<'PYEOF'
import json
import os
from pathlib import Path

report = Path(os.environ["FRESHNESS"])
coverage = ""
if report.is_file():
    coverage = json.loads(report.read_text(encoding="utf-8")).get("coverage_end_utc", "")
print(coverage or os.environ["FALLBACK"])
PYEOF
}
