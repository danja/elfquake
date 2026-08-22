#!/usr/bin/env bash
set -euo pipefail

. "$(dirname "$0")/lib/staleness.sh"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INPUT="${INPUT:-data/derived/models/all_italy_spatial_vlf_image_windows_aligned_windows.csv}"
OUT_DIR="${OUT_DIR:-data/derived/models/all_italy_spatial_permutation_controls}"
SEEDS="${SEEDS:-101 202 303 404 505}"
# Optional, as in evaluate-italy-spatial-baseline.sh: `target_cell_id` adds the
# cell-stratified metric so a shuffled control can be compared with a real run
# on the same metric rather than only on the pooled one.
STRATIFY_FIELD="${STRATIFY_FIELD:-}"

STRATIFY_ARGS=()
if [[ -n "$STRATIFY_FIELD" ]]; then STRATIFY_ARGS=(--stratify-field "$STRATIFY_FIELD"); fi

if [[ ! -f "$INPUT" ]]; then
  ./scripts/prepare-italy-spatial-model-inputs.sh
else
  # Item 89 replaced the existence-only rebuild guard inside
  # prepare-italy-spatial-model-inputs.sh, but every caller kept its own, so a
  # refreshed labeled table still stopped here: the aligned dataset exists, the
  # branch is skipped, and the evaluation reports a four-day-old record with
  # today's timestamp. Rebuild when missing, warn when merely stale.
  require_fresh_inputs "${SPATIAL_LABELS:-data/derived/multimodal/all_italy.spatial_vlf_image_windows.labeled.csv}" "$INPUT"
fi
mkdir -p "$OUT_DIR"

for SEED in $SEEDS; do
  PERMUTED="$OUT_DIR/permuted_${SEED}.csv"
  REPORT="$OUT_DIR/holdout_${SEED}.json"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli permute-spatial-targets \
    --input "$INPUT" --out "$PERMUTED" --seed "$SEED"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-temporal-holdout \
    --input "$PERMUTED" --out "$REPORT" --train-fraction 0.8 --epochs "${EPOCHS:-100}" \
    --learning-rate "${LEARNING_RATE:-0.2}" --group-by-time "${STRATIFY_ARGS[@]}"
done

printf 'permutation reports: %s\n' "$OUT_DIR"
