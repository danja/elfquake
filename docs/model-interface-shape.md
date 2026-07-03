# Model Interface Shape

Purpose: keep model inputs explicit before training multimodal or synthetic models.

Current audit artifact:

* `data/derived/models/interface_shape_audit.json`
* `data/derived/models/current_interface_alignment_manifest.json`

Run:

```sh
PYTHONPATH=src python -m elfquake.cli audit-model-interfaces \
  --input data/derived/ingv/events_italy_2026-06-01_2026-06-29.normalized.csv \
  --input data/derived/multimodal/cumiana_last_E_VLF.image_features.csv \
  --input data/derived/sim/mountain_256x256_seed42_10000.piezo.csv \
  --input data/derived/sim/mountain_256x256_seed42_10000.avalanche_signal.csv \
  --input data/derived/sim/mountain_256x256_seed42_10000.avalanche_events.csv \
  --input data/derived/sim/mountain_256x256_seed42_10000.summary.csv \
  --out data/derived/models/interface_shape_audit.json
```

## Current Shapes

| Shape | Examples | Current fit |
| --- | --- | --- |
| Event list | INGV normalized events, synthetic avalanche events | Aggregate to regular windows, or use a later event-process adapter |
| Image feature table | Cumiana VLF image features | Fits current `values.csv` + `masks.csv` + `index.csv` materializer |
| Sensor time series | `*.piezo.csv`, `*.avalanche_signal.csv` | Needs sequence materializer with `time x sensor x channel` axes |
| Summary time series | simulation `*.summary.csv` | Needs sequence materializer with `time x channel` axes |

## Interface Decision

Use two adapter families before target models:

1. Window adapter: turns real and synthetic seismic event lists into fixed region-time windows.
2. Sequence adapter: turns VLF-like and simulation sensor series into regular `time, entity, channel` tensors with matching present masks.

The current flat materializer remains useful for tabular/window features and VLF image summaries, but it is not sufficient for raw multimodal sequence models.

## Missing Data And Ablation

Every materialized numeric channel should have a matching present mask. Ablation should be declared by modality groups:

* seismic event/window features
* VLF image or VLF-derived sequence features
* astronomy and geomagnetic features
* synthetic piezo/VLF analogue features
* synthetic direct avalanche/seismic analogue features

Do not mix piezo/VLF and direct avalanche/seismic channels under one ablation switch.

## Current Adapter Artifacts

Event-window tables:

* `data/derived/models/ingv_italy_2026-06-01_2026-06-30.daily_event_windows.csv`
* `data/derived/models/mountain_256x256_seed42_10000.daily_synthetic_seismic_windows.csv`

Flat window tensor manifests:

* `data/derived/models/ingv_italy_2026-06-01_2026-06-30_daily_event_windows_tensor/manifest.json`
* `data/derived/models/mountain_256x256_seed42_10000_daily_synthetic_seismic_windows_tensor/manifest.json`

Sequence tensor manifests:

* `data/derived/models/mountain_256x256_seed42_10000_piezo_sequence/manifest.json`
* `data/derived/models/mountain_256x256_seed42_10000_avalanche_sequence/manifest.json`
* `data/derived/models/mountain_256x256_seed42_10000_summary_sequence/manifest.json`

Sequence manifests point to separate `time_axis.csv` and `entity_axis.csv` files so large runs do not bloat manifest JSON.

Aligned window datasets:

* `data/derived/models/mountain_256x256_seed42_10000.aligned_synthetic_windows.csv`
* `data/derived/models/mountain_256x256_seed42_10000_aligned_synthetic_windows_tensor/manifest.json`
* `data/derived/models/mountain_256x256_seed42_10000.aligned_hourly_synthetic_windows_gt1.csv`
* `data/derived/models/mountain_256x256_seed42_10000_aligned_hourly_synthetic_windows_gt1_tensor/manifest.json`
* `data/derived/models/ingv_italy_2026-06-01_2026-06-30.aligned_real_windows.csv`
* `data/derived/models/ingv_italy_2026-06-01_2026-06-30_aligned_real_windows_tensor/manifest.json`

The synthetic aligned table uses next-window synthetic event count as a smoke target. The current real aligned table is unlabeled and waits for target maturation.

The hourly synthetic `gt1` table is the current best smoke target:

* `167` labeled hourly rows
* target is next-hour synthetic event count greater than `1`
* chronological test fold has both classes

Synthetic smoke reports:

* `data/derived/models/mountain_256x256_seed42_10000.aligned_synthetic_windows.logistic_smoke.json`
* `data/derived/models/mountain_256x256_seed42_10000.aligned_synthetic_windows.ablation_smoke.json`
* `data/derived/models/mountain_256x256_seed42_10000.aligned_synthetic_windows.temporal_holdout.json`
* `data/derived/models/mountain_256x256_seed42_10000.aligned_hourly_synthetic_windows_gt1.temporal_holdout.json`

The temporal holdout trains on earlier rows and tests on later rows. The current six-row synthetic sample is only a pipeline check; its test fold contains only positive labels.

For the hourly `gt1` table, multimodal variants still overpredict positives and recover almost no true negatives. Treat this as a useful evaluator sanity check, not model evidence.

The temporal holdout report includes naive baselines and train-calibrated probability thresholds. Current `gt1` balanced accuracy is close to the naive `0.5` baseline, so the present single-seed synthetic data should not be used for model claims.

Alignment manifest:

* `data/derived/models/current_interface_alignment_manifest.json`

This links the current window tensors and sequence tensors into one model-run contract. Its ablation groups keep `seismic`, `vlf_image`, `synthetic_seismic`, `synthetic_piezo_vlf`, and `synthetic_direct_avalanche` separate.

Current caveats:

* Simulation sequence time coverage uses the declared synthetic mapping `2026-01-01T00:00:00Z` plus `60` seconds per step. This is a modeling assumption, not a physical calibration.

VLF image tensors now use explicit `vlf_image_captured_at_utc` index fields while preserving `vlf_image_source_file` provenance.
