#!/usr/bin/env bash
# Fetch and normalize the geomagnetic and solar index archives.
#
# Deliberately separate from refresh-prospective-labels.sh. The Kp/ap and F10.7
# archives are whole-history files of roughly 16 MB and 2 MB; the prospective
# timer runs every 30 minutes, and refetching those on that cadence would be
# both wasteful locally and abusive upstream. Run this daily instead, and let
# the prospective job read the normalized tables it leaves in
# data/derived/astronomy/.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RAW_ROOT="${SPACE_WEATHER_RAW_ROOT:-data/raw/astronomy}"
OUT_DIR="${SPACE_WEATHER_DERIVED_ROOT:-data/derived/astronomy}"
# Kyoto revises realtime Dst, so the previous month is refetched alongside the
# current one rather than trusted from the first pull.
DST_MONTHS="${DST_MONTHS:-$(date -u -d "-1 month" +%Y%m) $(date -u +%Y%m)}"
DST_TIER="${DST_TIER:-realtime}"
# Skip refetching a whole-history archive when a copy this new is already on
# disk. Both sources update once a day at most.
ARCHIVE_MAX_AGE_HOURS="${ARCHIVE_MAX_AGE_HOURS:-20}"

run_cli() {
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli "$@"
}

# Newest raw capture matching a filename prefix, or empty if there is none.
latest_raw() {
  find "$RAW_ROOT/captures" -type f -name "$1*" ! -name "*.metadata.json" 2>/dev/null \
    | sort | tail -n 1
}

fetch_if_stale() {
  local prefix="$1"
  local command="$2"
  local existing
  existing="$(latest_raw "$prefix")"
  if [[ -n "$existing" ]] && [[ -z "$(find "$existing" -mmin "+$((ARCHIVE_MAX_AGE_HOURS * 60))")" ]]; then
    echo "skip $command: $existing is under ${ARCHIVE_MAX_AGE_HOURS}h old"
    return
  fi
  run_cli "$command" --out-root "$RAW_ROOT"
}

fetch_if_stale gfz_kp_ap_since_1932 fetch-gfz-kp-ap
fetch_if_stale spaceweather_canada_f107_daily fetch-f107-daily

for month in $DST_MONTHS; do
  run_cli fetch-kyoto-dst --year-month "$month" --tier "$DST_TIER" --out-root "$RAW_ROOT"
done

KP_RAW="$(latest_raw gfz_kp_ap_since_1932)"
F107_RAW="$(latest_raw spaceweather_canada_f107_daily)"
if [[ -z "$KP_RAW" || -z "$F107_RAW" ]]; then
  echo "No Kp/ap or F10.7 raw capture found under $RAW_ROOT/captures" >&2
  exit 2
fi

run_cli normalize-gfz-kp-ap --raw "$KP_RAW" --out "$OUT_DIR/gfz_kp_ap.csv"
run_cli normalize-f107-daily --raw "$F107_RAW" --out "$OUT_DIR/f107_daily.csv"

# Normalize every Dst month that has ever been captured, not only this run's,
# so the derived layer keeps the full history rather than the last two months.
for month in $(find "$RAW_ROOT/captures" -type f -name "kyoto_dst_*_*.html" \
    | sed -E 's/.*kyoto_dst_[a-z]+_([0-9]{6})_.*/\1/' | sort -u); do
  raw="$(find "$RAW_ROOT/captures" -type f -name "kyoto_dst_*_${month}_*.html" | sort | tail -n 1)"
  run_cli normalize-kyoto-dst --raw "$raw" --out "$OUT_DIR/kyoto_dst_${month}.csv"
done

echo "space weather tables in $OUT_DIR"
