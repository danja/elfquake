#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUT="${OUT:-data/derived/models/piezo_group_holdout_comparison.json}"
REPORTS="${REPORTS:-data/derived/models/piezo_group_holdout/evaluation.json data/derived/models/piezo_group_holdout_h3/evaluation.json data/derived/models/piezo_group_holdout_h6_l60/evaluation.json data/derived/models/piezo_group_holdout_spatial/evaluation.json}"

report_args=()
for report in $REPORTS; do
  if [[ ! -f "$report" ]]; then
    echo "missing group holdout report: $report" >&2
    exit 1
  fi
  report_args+=(--report "$report")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli compare-piezo-group-holdouts \
  "${report_args[@]}" \
  --out "$OUT" \
  --balanced-accuracy-floor "${BALANCED_ACCURACY_FLOOR:-0.60}" \
  --recall-floor "${RECALL_FLOOR:-0.40}" \
  --fold-pass-fraction "${FOLD_PASS_FRACTION:-0.80}"
