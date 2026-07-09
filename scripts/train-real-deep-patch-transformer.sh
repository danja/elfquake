#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SCOPE="${SCOPE:-all_italy}"
ROOT="${ROOT:-data/derived/models/deep_patch_transformer}"
INPUT="${INPUT:-data/derived/models/${SCOPE}.real_vlf_aligned_windows.csv}"
READINESS="${READINESS:-data/derived/models/${SCOPE}.real_vlf_aligned_windows.readiness.json}"
SEQUENCE="${SEQUENCE:-data/derived/models/cumiana_vlf_image_sequence/manifest.json}"
SYNTHETIC_CHECKPOINT="${SYNTHETIC_CHECKPOINT:-$ROOT/deep_patch_transformer_synthetic.pt}"
SPLIT_INPUT="${SPLIT_INPUT:-$ROOT/${SCOPE}.real_finetune_split.csv}"
OUT="${OUT:-$ROOT/${SCOPE}.real_finetune.json}"
FINE_TUNED_CHECKPOINT="${FINE_TUNED_CHECKPOINT:-$ROOT/${SCOPE}.real_finetuned.pt}"

LOOKBACK_STEPS="${LOOKBACK_STEPS:-24}"
PATCH_STEPS="${PATCH_STEPS:-6}"
D_MODEL="${D_MODEL:-64}"
LAYERS="${LAYERS:-3}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"

mkdir -p "$ROOT"

if [[ ! -f "$READINESS" ]]; then
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-readiness \
    --input "$INPUT" \
    --out "$READINESS"
fi

STATUS="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status",""))' "$READINESS")"
if [[ "$STATUS" != "ready_for_smoke_training" ]]; then
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -c '
import json
import sys
from pathlib import Path

readiness = json.load(open(sys.argv[1], encoding="utf-8"))
out = Path(sys.argv[2])
report = {
    "schema": "elfquake.real_deep_patch_transformer_finetune.v1",
    "status": "blocked_insufficient_real_class_variation",
    "scope": sys.argv[3],
    "input": sys.argv[4],
    "sequence_manifest": sys.argv[5],
    "synthetic_checkpoint": sys.argv[6],
    "readiness_path": sys.argv[1],
    "labeled_row_count": readiness.get("labeled_row_count", 0),
    "positive_count": readiness.get("positive_count", 0),
    "negative_count": readiness.get("negative_count", 0),
    "source_status": readiness.get("status", ""),
    "strategy": "fine-tune from the synthetic patch Transformer checkpoint after real labels contain both classes",
}
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(
    "real deep patch Transformer blocked: {} positive / {} negative".format(
        report["positive_count"],
        report["negative_count"],
    )
)
print(f"output: {out}")
' "$READINESS" "$OUT" "$SCOPE" "$INPUT" "$SEQUENCE" "$SYNTHETIC_CHECKPOINT"
  exit 0
fi

PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -c '
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
out = Path(sys.argv[2])
time_field = sys.argv[3]
train_fraction = float(sys.argv[4])

with source.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fieldnames = list(reader.fieldnames or [])

labeled = [row for row in rows if row.get("target_occurred") in {"0", "1"}]
labeled.sort(key=lambda row: row.get(time_field, ""))
split_at = max(1, min(len(labeled) - 1, int(len(labeled) * train_fraction)))
labeled_ids = {id(row): index for index, row in enumerate(labeled)}
if "model_split" not in fieldnames:
    fieldnames.append("model_split")
for row in rows:
    index = labeled_ids.get(id(row))
    if index is None:
        row["model_split"] = ""
    else:
        row["model_split"] = "train" if index < split_at else "test"

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote chronological split: {split_at} train / {len(labeled) - split_at} test")
print(f"output: {out}")
' "$INPUT" "$SPLIT_INPUT" "${TIME_FIELD:-window_start_utc}" "${TRAIN_FRACTION:-0.8}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli train-torch-patch-transformer-split-holdout \
  --input "$SPLIT_INPUT" \
  --sequence-manifest "$SEQUENCE" \
  --out "$OUT" \
  --lookback-steps "$LOOKBACK_STEPS" \
  --patch-steps "$PATCH_STEPS" \
  --epochs "$EPOCHS" \
  --learning-rate "${LEARNING_RATE:-0.0005}" \
  --d-model "$D_MODEL" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --dropout "$DROPOUT" \
  --batch-size "$BATCH_SIZE" \
  --seed "${SEED:-42}" \
  --evaluation sequence_real_vlf_image_only \
  --checkpoint-in "$SYNTHETIC_CHECKPOINT" \
  --checkpoint-out "$FINE_TUNED_CHECKPOINT"
