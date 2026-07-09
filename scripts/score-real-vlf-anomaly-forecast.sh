#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ROOT:-data/derived/models/self_supervised}"
SEQUENCE="${SEQUENCE:-data/derived/models/cumiana_vlf_image_sequence/manifest.json}"
OUT="${OUT:-$ROOT/real_vlf_anomaly_forecast.json}"
SCORES_OUT="${SCORES_OUT:-$ROOT/real_vlf_anomaly_scores.csv}"

mkdir -p "$ROOT"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli score-sequence-anomalies \
  --sequence-manifest "$SEQUENCE" \
  --out "$OUT" \
  --scores-out "$SCORES_OUT" \
  --modality "${MODALITY:-real_vlf_image}" \
  --descriptor-profile "${DESCRIPTOR_PROFILE:-shape}" \
  --lookback-steps "${LOOKBACK_STEPS:-24}" \
  --stride "${STRIDE:-1}" \
  --train-fraction "${TRAIN_FRACTION:-0.8}" \
  --forecast-horizon-days "${FORECAST_HORIZON_DAYS:-7}" \
  --alert-threshold "${ALERT_THRESHOLD:-0.8}" \
  --mask-probability "${MASK_PROBABILITY:-0.15}" \
  --clean-loss-weight "${CLEAN_LOSS_WEIGHT:-0.0}" \
  --epochs "${EPOCHS:-30}" \
  --learning-rate "${LEARNING_RATE:-0.0003}" \
  --hidden-units "${HIDDEN_UNITS:-32}" \
  --embedding-units "${EMBEDDING_UNITS:-8}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --seed "${SEED:-42}"
