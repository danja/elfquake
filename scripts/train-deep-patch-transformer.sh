#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ROOT:-data/derived/models/deep_patch_transformer}"
SYNTHETIC_INPUT="${SYNTHETIC_INPUT:-data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv}"
REAL_SCOPE="${REAL_SCOPE:-all_italy}"
REAL_INPUT="${REAL_INPUT:-data/derived/models/${REAL_SCOPE}.real_vlf_aligned_windows.csv}"
REAL_READINESS="${REAL_READINESS:-data/derived/models/${REAL_SCOPE}.real_vlf_aligned_windows.readiness.json}"
REAL_SEQUENCE="${REAL_SEQUENCE:-data/derived/models/cumiana_vlf_image_sequence/manifest.json}"
REGIME_INPUT="${REGIME_INPUT:-$ROOT/synthetic.post_burn_in_regimes.csv}"
REGIME_REPORT="${REGIME_REPORT:-$ROOT/synthetic_regimes.json}"
SPLIT_INPUT="${SPLIT_INPUT:-$ROOT/synthetic.regime_balanced_split.csv}"
SPLIT_REPORT="${SPLIT_REPORT:-$ROOT/regime_balanced_split.json}"
SYNTHETIC_OUT="${SYNTHETIC_OUT:-$ROOT/deep_patch_transformer_synthetic.json}"
SYNTHETIC_CHECKPOINT="${SYNTHETIC_CHECKPOINT:-$ROOT/deep_patch_transformer_synthetic.pt}"
SUMMARY="${SUMMARY:-$ROOT/deep_patch_transformer_model_run_summary.json}"
REAL_OUT="${REAL_OUT:-$ROOT/${REAL_SCOPE}.real_finetune_readiness.json}"

LOOKBACK_STEPS="${LOOKBACK_STEPS:-120}"
PATCH_STEPS="${PATCH_STEPS:-12}"
D_MODEL="${D_MODEL:-64}"
LAYERS="${LAYERS:-3}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-24}"
BATCH_SIZE="${BATCH_SIZE:-32}"

mkdir -p "$ROOT"

manifest_args=()
for seed in 40 41 42; do
  manifest_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_avalanche_sequence/manifest.json")
  manifest_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_piezo_sequence/manifest.json")
  manifest_args+=(--sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_20000_summary_sequence/manifest.json")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli annotate-synthetic-regimes \
  --input "$SYNTHETIC_INPUT" \
  --out "$REGIME_INPUT" \
  --report "$REGIME_REPORT" \
  --regime-count "${REGIME_COUNT:-5}" \
  --burn-in-fraction "${BURN_IN_FRACTION:-0.2}" \
  --drop-burn-in

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli assign-balanced-split \
  --input "$REGIME_INPUT" \
  --out "$SPLIT_INPUT" \
  --report "$SPLIT_REPORT" \
  --group-field synthetic_regime_id \
  --test-fraction "${TEST_FRACTION:-0.2}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli train-torch-patch-transformer-split-holdout \
  --input "$SPLIT_INPUT" \
  "${manifest_args[@]}" \
  --out "$SYNTHETIC_OUT" \
  --lookback-steps "$LOOKBACK_STEPS" \
  --patch-steps "$PATCH_STEPS" \
  --epochs "$EPOCHS" \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --d-model "$D_MODEL" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --dropout "$DROPOUT" \
  --batch-size "$BATCH_SIZE" \
  --seed "${SEED:-42}" \
  --evaluation sequence_direct_avalanche_only \
  --evaluation sequence_piezo_vlf_only \
  --evaluation sequence_full \
  --checkpoint-out "$SYNTHETIC_CHECKPOINT"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-run-reports \
  --report "$SYNTHETIC_OUT" \
  --out "$SUMMARY"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -c '
import json
import sys
from pathlib import Path

readiness_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
real_input = sys.argv[3]
real_sequence = sys.argv[4]
readiness = json.loads(readiness_path.read_text(encoding="utf-8")) if readiness_path.exists() else {}
positive = int(readiness.get("positive_count", 0) or 0)
negative = int(readiness.get("negative_count", 0) or 0)
status = "ready_for_real_finetune" if positive and negative else "blocked_insufficient_real_class_variation"
report = {
    "schema": "elfquake.deep_patch_transformer_real_finetune_readiness.v1",
    "status": status,
    "real_input": real_input,
    "real_sequence_manifest": real_sequence,
    "synthetic_checkpoint": sys.argv[5],
    "readiness_path": str(readiness_path),
    "labeled_row_count": readiness.get("labeled_row_count", 0),
    "positive_count": positive,
    "negative_count": negative,
    "strategy": "pretrain patch Transformer on synthetic sequence modalities; fine-tune on real VLF sequence plus real aligned labels once both classes exist",
}
out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"real fine-tune status: {status}")
print(f"output: {out_path}")
' "$REAL_READINESS" "$REAL_OUT" "$REAL_INPUT" "$REAL_SEQUENCE" "$SYNTHETIC_CHECKPOINT"
