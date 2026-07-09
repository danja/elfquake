#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SCOPE="${SCOPE:-all_italy}"
INPUT="${INPUT:-data/derived/models/${SCOPE}.real_vlf_aligned_windows.csv}"
READINESS="${READINESS:-data/derived/models/${SCOPE}.real_vlf_aligned_windows.readiness.json}"
OUT="${OUT:-data/derived/models/${SCOPE}.real_vlf_aligned_windows.torch_tabular.json}"

if [[ ! -f "$READINESS" ]]; then
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-readiness \
    --input "$INPUT" \
    --out "$READINESS"
fi

STATUS="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status",""))' "$READINESS")"
if [[ "$STATUS" != "ready_for_smoke_training" ]]; then
  "$PYTHON_BIN" -c 'import json,sys
report=json.load(open(sys.argv[1], encoding="utf-8"))
print("refusing real training: status={}".format(report.get("status", "")))
print("rows={} labeled={} positive={} negative={}".format(report.get("row_count", 0), report.get("labeled_row_count", 0), report.get("positive_count", 0), report.get("negative_count", 0)))
' "$READINESS"
  exit 2
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli train-torch-tabular-holdout \
  --input "$INPUT" \
  --out "$OUT" \
  --train-fraction 0.8 \
  --epochs "${EPOCHS:-80}" \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --hidden-units "${HIDDEN_UNITS:-32}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --seed "${SEED:-42}"
