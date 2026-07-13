#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ROOT:-data/derived/models/damage_precursor_head}"
TARGET="${TARGET:-data/derived/models/damage_short_horizon/targets.csv}"
SEEDS="${SEEDS:-12300 12301 12302 12400 12401 12402 12500 12501 12502}"

mkdir -p "$ROOT"
args=()
for seed in $SEEDS; do
  args+=(--piezo-sequence-manifest "data/derived/models/mountain_256x256_seed${seed}_3000_piezo_sequence/manifest.json")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-damage-precursor-head \
  --target "$TARGET" "${args[@]}" --out "$ROOT/evaluation.json" \
  --short-steps "${SHORT_STEPS:-5}" --long-steps "${LONG_STEPS:-30}" \
  --epochs "${EPOCHS:-400}" --learning-rate "${LEARNING_RATE:-0.05}" --l2 "${L2:-0.01}"
