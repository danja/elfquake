#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MANIFEST="${MANIFEST:-data/raw/vlf/japan/manifest.csv}"
OUT_ROOT="${OUT_ROOT:-data/raw/vlf/japan}"
ONLY_ARGS=()
if [[ -n "${ONLY:-}" ]]; then ONLY_ARGS+=(--only "$ONLY"); fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli capture-vlf-japan-loop \
  --manifest "$MANIFEST" --out-root "$OUT_ROOT" \
  --cycles "${CYCLES:-0}" --interval-seconds "${INTERVAL_SECONDS:-1800}" \
  "${ONLY_ARGS[@]}"
