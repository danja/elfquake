#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUTPUT="${OUTPUT:-data/derived/models/common_transformer_fixture.csv}"
REPORT="${REPORT:-data/derived/models/common_transformer_fixture.json}"

# Channels that are constant across this corpus by construction, named here so
# the channel gate passes them deliberately rather than by inference. Each is
# uninformative and contributes nothing to a model; they are kept only so the
# synthetic and Japan tables keep their published schemas.
#
#   *_sample_count            fixed window length, identical for every window
#   relaxation_converged_*    the sandpile relaxation converges on every step
#   unstable_cell_count_*     no unstable cells survive relaxation, by design
#   safety_released_mass_*    the safety valve never fires in these episodes
#   japan_ch*_valid_fraction  all 9 Japan windows are fully valid
ALLOWED_CONSTANT_CHANNELS=(
  synthetic_piezo_vlf_sample_count
  synthetic_direct_avalanche_sample_count
  synthetic_summary_sample_count
  synthetic_summary_avalanche_count_max
  synthetic_summary_relaxation_converged_mean
  synthetic_summary_relaxation_converged_max
  synthetic_summary_relaxation_converged_sum
  synthetic_summary_unstable_cell_count_mean
  synthetic_summary_unstable_cell_count_max
  synthetic_summary_unstable_cell_count_sum
  synthetic_summary_safety_released_mass_mean
  synthetic_summary_safety_released_mass_max
  synthetic_summary_safety_released_mass_sum
  japan_ch1_valid_fraction_mean
  japan_ch1_valid_fraction_std
  japan_ch1_valid_fraction_max
  japan_ch2_valid_fraction_mean
  japan_ch2_valid_fraction_std
  japan_ch2_valid_fraction_max
)

allow_args=()
for channel in "${ALLOWED_CONSTANT_CHANNELS[@]}"; do
  allow_args+=(--allow-constant-channel "$channel")
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON_BIN" -m elfquake.cli build-common-window-fixture \
  --input data/derived/models/mountain_256x256_seed40_20000.aligned_hourly_synthetic_windows.csv \
  --input data/derived/multimodal/all_italy.spatial_vlf_image_windows.labeled.csv \
  --input data/derived/models/japan_vlf_model_input.m5.csv \
  --dataset-id seed40 \
  --dataset-id italy_all \
  --dataset-id japan_moshiri \
  "${allow_args[@]}" \
  --out "$OUTPUT" --report "$REPORT"
