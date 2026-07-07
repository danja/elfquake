#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SCOPES="${SCOPES:-all_italy central_italy}"
REAL_VLF_SEQUENCE="${REAL_VLF_SEQUENCE:-data/derived/models/cumiana_vlf_image_sequence/manifest.json}"

if [[ ! -f "$REAL_VLF_SEQUENCE" ]]; then
  ./materialize-real-vlf-sequence.sh
fi

for SCOPE in $SCOPES; do
  INPUT="data/derived/multimodal/${SCOPE}.prospective_vlf_image_windows.labeled.csv"
  SPEC="data/derived/models/${SCOPE}.prospective_vlf_image_windows_tensor_spec.json"
  TENSOR_DIR="data/derived/models/${SCOPE}_prospective_vlf_image_windows_tensor"
  ALIGNMENT="data/derived/models/${SCOPE}.real_vlf_alignment_manifest.json"
  ALIGNED="data/derived/models/${SCOPE}.real_vlf_aligned_windows.csv"
  READINESS="data/derived/models/${SCOPE}.real_vlf_aligned_windows.readiness.json"

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-tensor-spec \
    --input "$INPUT" \
    --out "$SPEC" \
    --time-field window_start_utc \
    --region-field region_id \
    --target-field target_occurred

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli materialize-tensor-dataset \
    --spec "$SPEC" \
    --out-dir "$TENSOR_DIR"

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-alignment-manifest \
    --manifest "$TENSOR_DIR/manifest.json" \
    --manifest "$REAL_VLF_SEQUENCE" \
    --run-id "${SCOPE}_real_vlf_prospective" \
    --out "$ALIGNMENT"

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-aligned-window-dataset \
    --base-manifest "$TENSOR_DIR/manifest.json" \
    --sequence-manifest "$REAL_VLF_SEQUENCE" \
    --out "$ALIGNED"

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli summarize-model-readiness \
    --input "$ALIGNED" \
    --out "$READINESS"
done
