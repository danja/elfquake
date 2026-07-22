#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${MANIFEST:-data/raw/vlf/japan/manifest.csv}"
STATION="${STATION:-mos}"
LOOKBACK_MONTHS="${LOOKBACK_MONTHS:-1}"
MAX_FILES="${MAX_FILES:-1}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

YEAR="${YEAR:-$(date -u -d "${LOOKBACK_MONTHS} months ago" +%Y)}"
MONTH="${MONTH:-$(date -u -d "${LOOKBACK_MONTHS} months ago" +%m)}"
ARCHIVE_URL="${ARCHIVE_URL:-https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/ground/vlf/${STATION}/${YEAR}/${MONTH}/}"

listing="$(mktemp)"
trap 'rm -f "$listing"' EXIT
curl --fail --location --max-time 60 --output "$listing" "$ARCHIVE_URL"

mapfile -t files < <(rg -o "isee_vlf_${STATION}_[0-9]+_v[0-9]+\\.cdf" "$listing" | sort -u | tail -n "$MAX_FILES")
[[ "${#files[@]}" -gt 0 ]] || { echo "No CDF files found at $ARCHIVE_URL" >&2; exit 2; }

for filename in "${files[@]}"; do
  url="${ARCHIVE_URL}${filename}"
  if rg -Fq ",$url," "$MANIFEST"; then
    echo "manifest exists: $filename"
    continue
  fi
  printf '%s\n' "isee_${STATION}_${filename#isee_vlf_${STATION}_},${url},cdf,Moshiri,44.37,142.27,crossed_loop,japan,application/cdf" >> "$MANIFEST"
  echo "manifest added: $url"
done
