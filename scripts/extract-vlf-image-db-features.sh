#!/usr/bin/env bash
# Palette-inverted absolute-dB features from Cumiana VLF spectrograms
# (next-actions item 97(a)).
#
# The pixel features in `cumiana_last_E_VLF.image_features.csv` are functions of
# colour, and the receiver's colour ramp moved by 11.58 dB during the July 2026
# outage, so the same colour means a different level either side of it. These
# features invert each capture through its own embedded colourbar instead, which
# is invariant to the palette setting by construction.
#
# Each band carries a censored fraction, and a band's level is withheld once
# censoring passes 50%: the two palettes resolve different dB windows, so a band
# below the shared floor is missing, not quiet, and a median over the surviving
# pixels would report the floor as a measurement.
#
# Being on one ruler is necessary for pooling the eras, not sufficient. After
# decoding, the late era still reads lower; whether that residual step is
# receiver gain or atmosphere is item 97(d) and is not settled here.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
IMAGE_ROOT="${IMAGE_ROOT:-data/raw/vlf/cumiana/captures}"
FILENAME_PREFIX="${FILENAME_PREFIX:-last_E_VLF}"
OUT="${OUT:-data/derived/multimodal/cumiana_last_E_VLF.image_db_features.csv}"

mkdir -p "$(dirname "$OUT")"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli extract-vlf-image-db-features \
  --image-root "$IMAGE_ROOT" \
  --filename-prefix "$FILENAME_PREFIX" \
  --residual-limit "${RESIDUAL_LIMIT:-30.0}" \
  --out "$OUT"

printf 'absolute-dB image features: %s\n' "$OUT"
