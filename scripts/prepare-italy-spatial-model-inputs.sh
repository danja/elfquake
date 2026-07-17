#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/multimodal/all_italy.spatial_vlf_image_windows.labeled.csv}"
PREFIX="${PREFIX:-data/derived/models/all_italy_spatial_vlf_image_windows}"
REAL_VLF_SEQUENCE="${REAL_VLF_SEQUENCE:-data/derived/models/cumiana_vlf_image_sequence/manifest.json}"

if [[ ! -f "$INPUT" ]]; then ./scripts/build-italy-spatial-vlf-targets.sh; fi
if [[ ! -f "$REAL_VLF_SEQUENCE" ]]; then ./scripts/materialize-real-vlf-sequence.sh; fi

SPEC="${PREFIX}_tensor_spec.json"
TENSOR_DIR="${PREFIX}_tensor"
ALIGNMENT="${PREFIX}_alignment_manifest.json"
ALIGNED="${PREFIX}_aligned_windows.csv"
READINESS="${PREFIX}_readiness.json"

run_cli() { PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli "$@"; }

run_cli build-tensor-spec --input "$INPUT" --out "$SPEC" \
  --time-field window_start_utc --region-field region_id --target-field target_occurred
run_cli materialize-tensor-dataset --spec "$SPEC" --out-dir "$TENSOR_DIR"
run_cli build-alignment-manifest --manifest "$TENSOR_DIR/manifest.json" \
  --manifest "$REAL_VLF_SEQUENCE" --run-id all_italy_spatial_real_vlf --out "$ALIGNMENT"
run_cli build-aligned-window-dataset --base-manifest "$TENSOR_DIR/manifest.json" \
  --sequence-manifest "$REAL_VLF_SEQUENCE" --out "$ALIGNED"
run_cli summarize-model-readiness --input "$ALIGNED" --out "$READINESS"

printf 'spatial input: %s\n' "$INPUT"
printf 'aligned dataset: %s\n' "$ALIGNED"
printf 'readiness report: %s\n' "$READINESS"
