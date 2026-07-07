# Analysis Report

Date: 2026-07-07

## Overview

ELFQuake is currently a feasibility pipeline, not an earthquake prediction system. The project can collect Italy-scoped INGV seismic events, Cumiana VLF spectrogram images, and astronomy/space-weather context; it can also generate synthetic seismic-like and VLF-like signals from an avalanche simulation. The latest useful work is on shaping those synthetic signals, checking whether PyTorch models can learn from multimodal synthetic data, and preparing real VLF-aligned model inputs. Real training remains blocked because the matured prospective real rows still contain only one target class per table.

## Scope

This report summarizes the current statistical comparison between real Italy-scoped seismic/VLF data and signals derived from the avalanche simulation, plus the current model-interface and PyTorch smoke results.

The comparison is diagnostic only. It does not demonstrate earthquake prediction capability.

## Inputs

Real seismic:

* `data/derived/ingv/events_italy_2026-06-01_2026-07-08.combined.normalized.csv`
* 175 normalized all-Italy INGV event rows
* `data/derived/ingv/events_central_italy_2026-06-01_2026-07-08.combined.normalized.csv`
* 22 normalized central-Italy INGV event rows

Real VLF:

* 247 Cumiana `last_E_VLF` spectrogram image-feature rows
* `data/derived/multimodal/cumiana_last_E_VLF.image_features.csv`
* image sequence manifest: `data/derived/models/cumiana_vlf_image_sequence/manifest.json`

Synthetic:

* current model training rows: `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv`
* 1005 combined hourly synthetic rows from seeds `40`, `41`, and `42`
* source simulation outputs include `*.piezo.csv`, `*.avalanche_signal.csv`, and `*.avalanche_events.csv`

Real prospective model rows:

* `data/derived/models/all_italy.real_vlf_aligned_windows.csv`
* `data/derived/models/central_italy.real_vlf_aligned_windows.csv`
* each table has 247 rows and 18 labeled rows
* all-Italy labels are currently 18 positive / 0 negative
* central-Italy labels are currently 0 positive / 18 negative
* both are `insufficient_class_variation`

## Method

Seismic event lists were converted into hourly binned energy traces using magnitude-derived energy proxies.

VLF spectrograms were reduced to column-intensity traces after cropping the plotted signal region.

Synthetic piezo and direct avalanche signals were reduced to per-step summed sensor traces.

For each trace, the report computes time-domain shape metrics and frequency-domain PSD metrics. Pairwise `normalized_distance` uses dimensionless shape and PSD-ratio metrics; amplitude, sample count, and duration are reported as deltas but are not included in that distance.

In model reports, `ablation` means removing one feature group or modality, such as VLF or astronomy, and comparing performance with and without it.

Output files:

* `data/derived/sim/mountain_256x256_seed42_10000.signal_shape_series.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.signal_shape_pairs.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.piezo_vlf_comparison.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.piezo_sensor_scan.csv`
* `data/derived/models/model_family_comparison.json`
* `data/derived/models/sequence_modality_diagnostic.json`

## Current Status

Real-data status:

* INGV refresh is working through `2026-07-08T00:00:00Z`.
* Cumiana VLF image capture and image-feature extraction are working.
* Real VLF-aligned model tables are scaffolded for all-Italy and central Italy.
* Real PyTorch training should not start yet, because each real table has only one target class.

Synthetic-model status:

* The current synthetic model table has 1005 rows and enough positive/negative labels for smoke modeling.
* CPU PyTorch tabular and GRU sequence models are implemented and compared.
* Best calibrated synthetic family row is `0.772558` balanced accuracy for `sequence_piezo_vlf_only` on held-out `seed42`.
* Full sequence sweep best calibrated row is `0.766942` for `sequence_direct_avalanche_only`, `lookback=60`, `hidden=24`, held-out `seed42`.
* The sequence diagnostic says not to change defaults yet: the prior best run used 20 epochs, the sweep used 10, temporal sequence rows stay near `0.5`, and mean group performance is highest for `sequence_full`.

## Shape Diagnostics

Real seismic vs synthetic seismic event traces:

* current seed `42` shape distance after sparse direct avalanche peak extraction: `1.5258`
* real seismic is sparse: nonzero ratio `0.1583`
* synthetic event trace is less dense than before: nonzero ratio `0.7652`
* real seismic PSD slope: `-0.0670`
* synthetic event PSD slope: `-0.1678`

Real VLF image columns vs synthetic piezo/VLF signal:

* shape distance: `1.4620`
* real VLF PSD slope: `-0.3506`
* synthetic piezo PSD slope: `-0.2705`
* real VLF lag-1 autocorrelation: `0.5381`
* synthetic piezo lag-1 autocorrelation: `0.9658`

VLF image-level comparison:

* nearest Cumiana image: `data/raw/vlf/cumiana/captures/2026-06-30/last_E_VLF_2026-06-30T23-15-00Z.jpg`
* nearest normalized distance after render tuning: `13.6492`
* simulated intensity mean: `0.5987`
* real mean intensity: `0.4687`
* simulated high-intensity ratio: `0.1695`
* real high-intensity ratio: `0.1712`
* simulated vertical streak count: `115`
* real mean vertical streak count: `117.6957`

Piezo sensor scan:

* all 16 piezo sensors were compared individually against the current Cumiana VLF image-column trace
* default pre-tuning best sensor: `9`
* tuned best sensor: `5`
* default pre-tuning best sensor lag-1 autocorrelation: `0.8081`
* tuned best sensor lag-1 autocorrelation: `0.6002`
* current VLF reference lag-1 autocorrelation in the scan: about `0.5930`
* summed piezo lag-1 autocorrelation remains much smoother at about `0.9658`
* corrected default pre-tuning shape score: `0.9701`
* tuned threshold/locality shape score: `0.6217`
* tuned burst-run rate: `0.0299`; current VLF reference rate differs by about `0.0073`
* experimental threshold-40 release alone improved lag, but adding receiver locality improved the overall corrected score

Piezo multi-seed validation:

* output CSV: `data/derived/sim/piezo_seed_validation_summary.csv`
* seed `40`: best sensor `10`, shape score `0.6504`, lag-1 `0.5816`
* seed `41`: best sensor `0`, shape score `0.6206`, lag-1 `0.6267`
* seed `42`: best sensor `5`, shape score `0.6165`, lag-1 `0.6002`
* mean shape score over seeds `40`-`42`: `0.6292`
* mean lag-1 autocorrelation over seeds `40`-`42`: `0.6028`

Full-size multi-seed check:

* `compare-simulation-grid.sh` ran for full-size `256 x 256`, `10000` step runs on seeds `40`, `41`, and `42`
* metrics-only runs used `RUN_HEATMAPS=0`, `RUN_VIDEO=0`, and `RUN_AUDIO=0`
* summary CSV: `data/derived/sim/full_size_seed_comparison.csv`

| seed | events | seismic distance | event nonzero | event PSD slope | piezo/VLF distance | piezo lag-1 | nearest VLF image distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 189 | 1.5775 | 0.8030 | 0.2524 | 1.4041 | 0.9581 | 13.2128 |
| 41 | 191 | 1.6198 | 0.8106 | -0.4042 | 1.4663 | 0.9666 | 13.4896 |
| 42 | 172 | 1.5258 | 0.7652 | -0.1678 | 1.4620 | 0.9658 | 13.6492 |

Direct avalanche event extraction tuning:

* tuning grid: quantiles `0.90`, `0.95`, `0.975`, `0.99`; local-max windows `5`, `15`, `30`, `60`
* output CSVs: `data/derived/sim/mountain_256x256_seed*_10000.avalanche_event_tuning.csv`
* best quantile for seeds `40`, `41`, and `42`: `0.99`
* best local-max window is seed-dependent: seed `40` prefers `5`, seed `41` prefers `60`, seed `42` prefers `30`
* current default `0.95/15` is denser than the tuned `0.99` candidates
* longer `20000` step runs for seeds `40`, `41`, and `42` also favour quantile `0.99`
* in the longer runs, window `30` is best for seeds `40` and `41`, and second-best for seed `42`

## Interpretation

Sparse local-peak extraction improved the direct synthetic seismic event trace. It is still denser than the real INGV trace, but the PSD slope and overall distance are much closer than the previous all-nonzero event extraction.

Across the full-size seed check, seed `42` is still the best current default on direct seismic event-shape distance and sparsity, but all tested seeds remain much denser than real seismic events. The next tuning target is therefore direct avalanche event extraction thresholds and burst spacing, not map projection.

The extraction tuning pass supports increasing the avalanche event quantile to `0.99` and local-max window to `30`. Window `30` is not best for every run, but it is the most stable current default across the longer multi-seed check.

The piezo/VLF image rendering is now much closer to Cumiana image statistics for brightness, high-intensity coverage, and vertical streaks. The underlying piezo time series is still too smooth in time, with very high autocorrelation, so later tuning should focus on signal dynamics rather than adding display artifacts.

The per-sensor scan shows that summing all piezo sensors over-smooths the VLF-like channel. Single sensors are more plausible. Raw burst-run counts were misleading because the image-derived VLF trace has many more samples than the simulation trace, so comparisons now use burst-run rate.

The current tuned default uses thresholded accumulated-charge release plus a local receiver footprint. It improves lag-1 autocorrelation and the corrected overall shape score without artificial spikes. The seed `40`-`42` validation is consistent enough to keep these as current simulation defaults, but the best receiver is seed-dependent, so model data preparation should preserve sensor identity and support sensor selection or pooling.

Piezo receiver locality is now separated from the direct avalanche signal range. Tuning the VLF-like piezo channel should not silently change the seismic-like avalanche channel.

After that separation, seed `40`-`42` simulation CSVs, sparse avalanche event lists, event maps, aligned synthetic model rows, tensors, and smoke reports were refreshed. The combined `gt0` aligned table still has `501` rows with `160` positives and `341` negatives. The chronological holdout remains weak: best default balanced accuracy is `0.5333` for `synthetic_seismic_piezo_vlf`. Leave-one-seed-out checks remain stronger, with best default balanced accuracy from `0.7177` to `0.7947` depending on the held-out seed.

Longer synthetic run check:

* regenerated seed `40`, `41`, and `42` at `20000` steps under current defaults
* sparse event counts: seed `40` has `130`, seed `41` has `129`, seed `42` has `135`
* combined `gt0` aligned table: `1005` rows, `364` positives, `641` negatives
* combined `gt1` aligned table: `1005` rows, `30` positives, `975` negatives
* `gt0` chronological best default balanced accuracy: `0.4833`
* `gt0` leave-one-seed-out best default balanced accuracy range: `0.6054` to `0.6284`
* `gt1` remains mostly a sparsity check

Chronological split diagnostics:

* diagnostic outputs:
  * `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.temporal_diagnostics.json`
  * `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.temporal_diagnostics.csv`
  * `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows_gt1.temporal_diagnostics.json`
  * `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows_gt1.temporal_diagnostics.csv`
* `gt0` train positive rate is `0.2935`; test positive rate is `0.6368`
* `gt1` train positive rate is `0.0249`; test positive rate is `0.0498`
* later test windows have much higher terrain/topple intensity: `synthetic_summary_max_height_mean`, `synthetic_direct_avalanche_active_topple_cell_count_*`, and `synthetic_summary_topple_count_*` all drift upward by about one training standard deviation
* several synthetic seismic aggregate features flip their target correlation sign between train and test in the `gt0` split

The longer run increases target support but does not improve chronological generalization. It should be used to stress-test the model interface and synthetic-transfer workflow, not as evidence of predictive value.

The weak `20000` chronological holdout appears to be mostly a non-stationary split problem. The simulation trajectory changes regime over time: later windows are taller and more avalanche-active, and the target rate changes substantially. Time-split evaluation is still the right conservative check, but the current synthetic generator needs burn-in handling, regime-balanced splitting, or longer/more varied runs before model metrics are meaningful.

The direct avalanche signal should remain separate from the piezo/VLF channel. Cross-modality distances can be useful sanity checks, but tuning should compare real seismic primarily with direct avalanche outputs, and real VLF primarily with piezo-derived outputs.

## Model Scaffold Update

Current CPU PyTorch tabular and sequence reports can now be compared directly with `compare-model-runs.sh`. The sequence path also has a bounded sweep script, a missing-modality smoke script, and a real Cumiana VLF image sequence materializer so synthetic piezo/VLF and real VLF inputs keep the same model-facing shape.

Smoke outputs:

* tabular-vs-sequence comparison: `data/derived/models/mountain_256x256_seeds40-42_20000.tabular_vs_sequence_model_comparison.json`
* best calibrated row in that comparison: `0.772558`, `sequence_piezo_vlf_only`, held-out `seed42`
* tiny sequence sweep smoke: `data/derived/models/sequence_sweep_smoke/sequence_sweep_comparison.json`; best calibrated row `0.709624`, `sequence_full`, held-out `seed41`
* missing-modality smoke: `data/derived/models/missing_modality/missing_modality_seed42_summary.json`; no-piezo direct avalanche scored higher than piezo-only in this short run
* real VLF image sequence manifest: `data/derived/models/cumiana_vlf_image_sequence/manifest.json`, with `247` time steps and `25` channels
* INGV refresh through `2026-07-08T00:00:00Z`: all-Italy prospective labels have `18` positives and `0` negatives; central Italy has `18` negatives and `0` positives, so real training still has insufficient class variation
* full sequence sweep: `data/derived/models/sequence_sweep/sequence_sweep_comparison.json`, `24` reports; best calibrated row `0.766942`, `sequence_direct_avalanche_only`, `lookback=60`, `hidden=24`, held-out `seed42`
* combined family comparison: `data/derived/models/model_family_comparison.json`, `37` rows; best calibrated row remains `0.772558`, `sequence_piezo_vlf_only`, held-out `seed42`
* sequence modality diagnostic: `data/derived/models/sequence_modality_diagnostic.json`, `112` evaluation rows; best default sequence row uses `20` epochs and piezo/VLF-only, while best sweep row uses `10` epochs and direct avalanche-only
* matched 20-epoch sequence comparison: `data/derived/models/sequence_sweep_20epoch/default_vs_matched_sequence_diagnostic.json`, `64` evaluation rows; best row remains `sequence_piezo_vlf_only`, `lookback=60`, `hidden=24`, held-out `seed42`, calibrated balanced accuracy `0.772558`
* repeated training-seed comparison: `data/derived/models/sequence_training_seed_repeat/sequence_training_seed_selection.json`; best single row is still `sequence_piezo_vlf_only` at `0.772558`, but `sequence_full` has the best mean group score (`0.741342`) and best worst-held-out-seed score (`0.712754`)
* real model-input scaffold: `data/derived/models/all_italy.real_vlf_aligned_windows.csv` and `data/derived/models/central_italy.real_vlf_aligned_windows.csv`; both have `247` rows and `18` labeled rows but still lack class variation

Sequence diagnostic interpretation: do not change the default GRU lookback from `60` on current evidence. Repeated training seeds confirm the strongest single row is still piezo/VLF-only on held-out `seed42`; however, `sequence_full` is more stable across held-out seeds and training seeds. All temporal sequence rows remain near balanced accuracy `0.5`, so these are still synthetic-transfer diagnostics rather than evidence of real predictive skill.

## Limitations

The real VLF data is image-derived, not raw waveform data.

The current real seismic sample covers June 1 through July 8, 2026 and is still far too short for robust statistical claims.

Simulation step time is an assumed mapping. Frequency-domain comparisons are therefore shape diagnostics, not physical frequency validation.

The current direct avalanche signal input uses `*.avalanche_signal.csv`; model smoke runs currently use the 20000-step seed `40`-`42` synthetic tables.
