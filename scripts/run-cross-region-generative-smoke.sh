#!/usr/bin/env bash
set -euo pipefail

# CPU-only engineering smoke test; Japan inputs remain research-use-only.
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ROOT:-data/derived/models/cross_region_generative_smoke}"
TARGET="${TARGET:-$ROOT/italy_target.csv}"
REPORT="${REPORT:-$ROOT/transformer_report.json}"
MAP_DIR="${MAP_DIR:-$ROOT/map_inputs}"
MAP="${MAP:-docs/images/cross-region-generative-smoke.png}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}"
SEED="${SEED:-42}"

mkdir -p "$ROOT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.models.cross_region_smoke \
  prepare-target \
  --source data/derived/models/common_transformer_fixture.csv \
  --events "$REAL_EVENTS" \
  --out "$TARGET"

manifest_args=(
  --synthetic-sequence-manifest data/derived/models/common_transformer_fixture_sequences/seed40_synthetic_direct_avalanche_sequence/manifest.json
  --synthetic-sequence-manifest data/derived/models/common_transformer_fixture_sequences/seed40_synthetic_piezo_vlf_sequence/manifest.json
  --synthetic-sequence-manifest data/derived/models/common_transformer_fixture_sequences/seed40_synthetic_summary_sequence/manifest.json
)
japan_args=(
  --japan-sequence-manifest data/derived/models/common_transformer_fixture_sequences/japan_moshiri_japan_vlf_sequence/manifest.json
)
italy_args=(
  --italy-sequence-manifest data/derived/models/common_transformer_fixture_sequences/italy_all_seismic_sequence/manifest.json
  --italy-sequence-manifest data/derived/models/common_transformer_fixture_sequences/italy_all_astronomy_sequence/manifest.json
)

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-self-supervised-transformer \
  --target "$TARGET" \
  "${manifest_args[@]}" \
  --real-sequence-manifest data/derived/models/common_transformer_fixture_sequences/italy_all_italy_vlf_sequence/manifest.json \
  "${japan_args[@]}" "${italy_args[@]}" \
  --target-dataset-id italy_all \
  --italy-modality seismic --italy-modality italy_vlf --italy-modality astronomy \
  --regime synthetic_then_japan_then_italy \
  --out "$REPORT" \
  --seed "$SEED" \
  --lookback-steps "${LOOKBACK_STEPS:-12}" \
  --patch-steps "${PATCH_STEPS:-3}" \
  --ssl-epochs "${SSL_EPOCHS:-3}" \
  --supervised-epochs "${SUPERVISED_EPOCHS:-6}" \
  --d-model "${D_MODEL:-32}" --layers "${LAYERS:-2}" --heads "${HEADS:-4}" \
  --batch-size "${BATCH_SIZE:-64}" --max-pretrain-windows "${MAX_PRETRAIN_WINDOWS:-512}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.models.cross_region_smoke \
  map-inputs --report "$REPORT" --target "$TARGET" --real-events "$REAL_EVENTS" --out-dir "$MAP_DIR"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli \
  render-transfer-trial-map \
  --actual-events "$MAP_DIR/heldout_week_actual_events.csv" \
  --predictions "$MAP_DIR/heldout_week_predictions.csv" \
  --out "$MAP" --metadata-out "${MAP%.png}.json" \
  --title "ELFQuake cross-region smoke test: actual vs generated event coordinates" \
  --prediction-label "generated event coordinate"

echo "report: $REPORT"
echo "map: $MAP"
