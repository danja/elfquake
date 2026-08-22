#!/usr/bin/env bash
set -euo pipefail

. "$(dirname "$0")/lib/staleness.sh"

# CPU-only engineering smoke test; Japan inputs remain research-use-only.
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ROOT="${ROOT:-data/derived/models/cross_region_generative_smoke}"
TARGET="${TARGET:-$ROOT/italy_target.csv}"
REPORT="${REPORT:-$ROOT/transformer_report.json}"
MAP_DIR="${MAP_DIR:-$ROOT/map_inputs}"
MAP="${MAP:-docs/images/cross-region-generative-smoke.png}"
REAL_EVENTS="${REAL_EVENTS:-data/derived/ingv/events_italy_all_available.combined.normalized.csv}"
SEED="${SEED:-42}"
FIXTURE="${FIXTURE:-data/derived/models/common_transformer_fixture.csv}"
SEQUENCE_ROOT="${SEQUENCE_ROOT:-data/derived/models/common_transformer_fixture_sequences}"

seq_manifest() { printf '%s/%s/manifest.json' "$SEQUENCE_ROOT" "$1"; }

SEQUENCES=(
  seed40_synthetic_direct_avalanche_sequence
  seed40_synthetic_piezo_vlf_sequence
  seed40_synthetic_summary_sequence
  japan_moshiri_japan_vlf_sequence
  italy_all_italy_vlf_sequence
  italy_all_seismic_sequence
  italy_all_astronomy_sequence
)

# Two hops, two chances to go stale. The fixture is built from the labeled
# spatial table and the sequences are materialized from the fixture, and
# neither step runs on any refresh path. On 2026-08-22 the sequences were
# 13 days behind the fixture and still carried the two dead astronomy
# channels of item 99 while the fixture already had 19 live ones.
#
# Check the seven manifests this run reads, not the whole sequence root:
# materialize-common-transformer-sequences.sh rewrites only the datasets the
# current fixture contains, so the root also holds 40 orphaned Japan per-file
# manifests from July that nothing here opens and nothing will ever refresh.
manifest_paths=()
for name in "${SEQUENCES[@]}"; do manifest_paths+=("$(seq_manifest "$name")"); done

require_fresh_inputs "$REAL_EVENTS" "$FIXTURE"
require_fresh_inputs "$FIXTURE" "${manifest_paths[@]}"

mkdir -p "$ROOT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.models.cross_region_smoke \
  prepare-target \
  --source "$FIXTURE" \
  --events "$REAL_EVENTS" \
  --out "$TARGET"

manifest_args=(
  --synthetic-sequence-manifest "$(seq_manifest seed40_synthetic_direct_avalanche_sequence)"
  --synthetic-sequence-manifest "$(seq_manifest seed40_synthetic_piezo_vlf_sequence)"
  --synthetic-sequence-manifest "$(seq_manifest seed40_synthetic_summary_sequence)"
)
japan_args=(
  --japan-sequence-manifest "$(seq_manifest japan_moshiri_japan_vlf_sequence)"
)
italy_args=(
  --italy-sequence-manifest "$(seq_manifest italy_all_seismic_sequence)"
  --italy-sequence-manifest "$(seq_manifest italy_all_astronomy_sequence)"
)

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli evaluate-self-supervised-transformer \
  --target "$TARGET" \
  "${manifest_args[@]}" \
  --real-sequence-manifest "$(seq_manifest italy_all_italy_vlf_sequence)" \
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
