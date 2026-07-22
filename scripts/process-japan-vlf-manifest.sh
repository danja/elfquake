#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MANIFEST="${MANIFEST:-data/raw/vlf/japan/manifest.csv}"
RAW_ROOT="${RAW_ROOT:-data/raw/vlf/japan}"
FEATURE_ROOT="${FEATURE_ROOT:-data/derived/vlf/japan}"
WINDOWS="${WINDOWS:-}"
FORCE="${FORCE:-0}"

mkdir -p "$RAW_ROOT" "$FEATURE_ROOT"

tail -n +2 "$MANIFEST" | while IFS=, read -r endpoint_id url content station latitude longitude receiver_mode region_id expected_content_type; do
  [[ -n "$endpoint_id" ]] || continue
  [[ "$content" == "cdf" ]] || { echo "skip ${endpoint_id}: content is not cdf"; continue; }

  filename="$(basename "${url%%\?*}")"
  raw="${RAW_ROOT}/${filename}"
  stem="${filename%.cdf}"
  metadata="${FEATURE_ROOT}/${stem}.metadata.json"
  features="${FEATURE_ROOT}/${stem}.features.csv"
  windows_out="${FEATURE_ROOT}/${stem}.window_features.csv"

  if [[ ! -s "$raw" ]]; then
    URL="$url" OUTPUT="$raw" ./scripts/fetch-japan-vlf-cdf.sh
  else
    echo "raw exists: $raw"
  fi
  if [[ "$FORCE" == "1" || ! -s "$metadata" ]]; then
    INPUT="$raw" PYTHON_BIN="$PYTHON_BIN" ./scripts/normalize-japan-vlf-cdf.sh
  else
    echo "metadata exists: $metadata"
  fi
  if [[ "$FORCE" == "1" || ! -s "$features" ]]; then
    INPUT="$raw" OUTPUT="$features" PYTHON_BIN="$PYTHON_BIN" ./scripts/extract-japan-vlf-cdf-features.sh
  else
    echo "features exist: $features"
  fi
  if [[ -n "$WINDOWS" ]]; then
    FEATURES="$features" WINDOWS="$WINDOWS" OUTPUT="$windows_out" PYTHON_BIN="$PYTHON_BIN" \
      ./scripts/build-japan-vlf-cdf-window-features.sh
  fi
  echo "processed ${endpoint_id} station=${station} region=${region_id} research_use_only=1"
done
