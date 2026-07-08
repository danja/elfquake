#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ROOT:-data/derived/models/self_supervised}"
SEQUENCE="${SEQUENCE:-data/derived/models/cumiana_vlf_image_sequence/manifest.json}"
OUT="${OUT:-$ROOT/real_vlf_image_autoencoder.json}"
CHECKPOINT_OUT="${CHECKPOINT_OUT:-$ROOT/real_vlf_image_autoencoder.pt}"
EMBEDDINGS_OUT="${EMBEDDINGS_OUT:-$ROOT/real_vlf_image_embeddings.csv}"

mkdir -p "$ROOT"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli pretrain-sequence-autoencoder \
  --sequence-manifest "$SEQUENCE" \
  --out "$OUT" \
  --modality "${MODALITY:-real_vlf_image}" \
  --lookback-steps "${LOOKBACK_STEPS:-24}" \
  --stride "${STRIDE:-1}" \
  --train-fraction "${TRAIN_FRACTION:-0.8}" \
  --mask-probability "${MASK_PROBABILITY:-0.15}" \
  --epochs "${EPOCHS:-40}" \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --hidden-units "${HIDDEN_UNITS:-64}" \
  --embedding-units "${EMBEDDING_UNITS:-16}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --seed "${SEED:-42}" \
  --checkpoint-out "$CHECKPOINT_OUT" \
  --embeddings-out "$EMBEDDINGS_OUT"
