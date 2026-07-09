#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ROOT:-data/derived/models/self_supervised}"
REAL_SEQUENCE="${REAL_SEQUENCE:-data/derived/models/cumiana_vlf_image_sequence/manifest.json}"
SEEDS="${SEEDS:-40 41 42}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
STEPS="${STEPS:-20000}"
OUT="${OUT:-$ROOT/real_vlf_mixed_domain_alignment.json}"
EMBEDDINGS_OUT="${EMBEDDINGS_OUT:-$ROOT/real_vlf_mixed_domain_alignment_embeddings.csv}"
CONTROL_METHODS="${CONTROL_METHODS:-centroid random full}"

mkdir -p "$ROOT"

manifest_args=()
for seed in $SEEDS; do
  manifest="data/derived/models/mountain_${WIDTH}x${HEIGHT}_seed${seed}_${STEPS}_piezo_sequence/manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "error: missing synthetic sequence manifest: $manifest" >&2
    exit 2
  fi
  manifest_args+=(--synthetic-sequence-manifest "$manifest")
done

control_args=()
for method in $CONTROL_METHODS; do
  control_args+=(--control-method "$method")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-mixed-domain-alignment \
  --real-sequence-manifest "$REAL_SEQUENCE" \
  "${manifest_args[@]}" \
  --out "$OUT" \
  --real-modality "${REAL_MODALITY:-real_vlf_image}" \
  --synthetic-modality "${SYNTHETIC_MODALITY:-synthetic_piezo_vlf}" \
  --descriptor-profile "${DESCRIPTOR_PROFILE:-shape}" \
  --lookback-steps "${LOOKBACK_STEPS:-24}" \
  --stride "${STRIDE:-1}" \
  --train-fraction "${TRAIN_FRACTION:-0.8}" \
  --mask-probability "${MASK_PROBABILITY:-0.15}" \
  --clean-loss-weight "${CLEAN_LOSS_WEIGHT:-0.0}" \
  --inlier-fraction "${INLIER_FRACTION:-0.25}" \
  --inlier-method "${INLIER_METHOD:-local}" \
  "${control_args[@]}" \
  --max-synthetic-train-windows "${MAX_SYNTHETIC_TRAIN_WINDOWS:-15000}" \
  --coral-weight "${CORAL_WEIGHT:-0.1}" \
  --epochs "${EPOCHS:-30}" \
  --learning-rate "${LEARNING_RATE:-0.0003}" \
  --hidden-units "${HIDDEN_UNITS:-32}" \
  --embedding-units "${EMBEDDING_UNITS:-8}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --seed "${SEED:-42}" \
  --embeddings-out "$EMBEDDINGS_OUT"
