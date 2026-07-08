# Analysis Report

Date: 2026-07-08

## Overview

ELFQuake is currently a feasibility pipeline, not an earthquake prediction system. The project can collect Italy-scoped INGV seismic events, Cumiana VLF spectrogram images, and astronomy/space-weather context; it can also generate synthetic seismic-like and VLF-like signals from an avalanche simulation. Historical seismic-only backfill now covers 2024-2026 and gives usable smoke baselines. Supervised real VLF target training remains blocked because the matured prospective VLF rows still contain only one target class per table, so the default modeling path is self-supervised real VLF pretraining plus a weak end-to-end trial forecast artifact that establishes the downstream event-list shape.

## Scope

This report summarizes the current statistical comparison between real Italy-scoped seismic/VLF data and signals derived from the avalanche simulation, plus the current model-interface and PyTorch smoke results.

The comparison is diagnostic only. It does not demonstrate earthquake prediction capability.

## Inputs

Real seismic:

* `data/derived/ingv/events_italy_2026-06-01_2026-07-08.combined.normalized.csv`
* 176 normalized all-Italy INGV event rows
* `data/derived/ingv/events_central_italy_2026-06-01_2026-07-08.combined.normalized.csv`
* 22 normalized central-Italy INGV event rows
* `data/derived/ingv/events_italy_all_available.combined.normalized.csv`
* 4836 normalized all-Italy rows, from `2024-01-01T21:38:30.320000Z` to `2026-07-07T08:51:55.110000Z`
* `data/derived/ingv/events_central_italy_all_available.combined.normalized.csv`
* 594 normalized central-Italy rows, from `2024-01-03T07:43:40.720000Z` to `2026-07-07T08:51:55.110000Z`

Real VLF:

* 247 Cumiana `last_E_VLF` spectrogram image-feature rows
* `data/derived/multimodal/cumiana_last_E_VLF.image_features.csv`
* image sequence manifest: `data/derived/models/cumiana_vlf_image_sequence/manifest.json`

Synthetic:

* current model training rows: `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv`
* 501 combined relabeled hourly synthetic rows from seeds `40`, `41`, and `42`
* source simulation outputs include `*.piezo.csv`, `*.avalanche_signal.csv`, and `*.avalanche_events.csv`

Real prospective model rows:

* `data/derived/models/all_italy.real_vlf_aligned_windows.csv`
* `data/derived/models/central_italy.real_vlf_aligned_windows.csv`
* each table has 247 rows and 54 labeled rows
* all-Italy labels are currently 55 positive / 0 negative
* central-Italy labels are currently 0 positive / 55 negative
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
* `data/derived/sim/mountain_256x256_seed42_10000.central_italy_signal_shape_series.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.central_italy_signal_shape_pairs.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.all_italy_signal_shape_series.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.all_italy_signal_shape_pairs.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.piezo_vlf_comparison.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.piezo_sensor_scan.csv`
* `data/derived/models/model_family_comparison.json`
* `data/derived/models/sequence_full_regime/sequence_full_model_run_summary.json`
* `data/derived/models/sequence_full_regime/post_burn_in_temporal_split_diagnostics.json`
* `data/derived/models/sequence_full_balanced/sequence_full_balanced_model_run_summary.json`
* `data/derived/models/sequence_full_balanced/regime_balanced_split.json`
* `data/derived/models/tiny_patch_transformer/tiny_patch_transformer_model_run_summary.json`
* `data/derived/models/real_synthetic_compact_comparison.json`
* `data/derived/models/deep_patch_transformer/deep_patch_transformer_synthetic.json`
* `data/derived/models/deep_patch_transformer/all_italy.real_finetune.json`
* `data/derived/models/deep_patch_transformer/central_italy.real_finetune.json`
* `data/derived/models/self_supervised/real_vlf_image_autoencoder.json`
* `data/derived/models/self_supervised/real_vlf_image_embeddings.csv`
* `data/derived/models/self_supervised/real_vlf_vs_synthetic_piezo_embedding_domain.json`
* `data/derived/models/self_supervised/real_vlf_vs_synthetic_piezo_embeddings.csv`
* `data/derived/models/trial_forecast/mag_gt2_weekly_trial_forecast.json`
* `data/derived/models/trial_forecast/mag_gt2_weekly_trial_events.csv`
* `data/derived/models/missing_modality/missing_modality_seed42_summary.json`
* `data/derived/models/sequence_modality_diagnostic.json`
* `data/derived/models/all_italy.ingv_backfill_seismic_windows.temporal_holdout.json`
* `data/derived/models/central_italy.ingv_backfill_seismic_windows.temporal_holdout.json`

## Current Status

Real-data status:

* INGV prospective refresh is working through `2026-07-08T00:00:00Z`; historical backfill currently runs through `2026-07-07T00:00:00Z`.
* Cumiana VLF image capture and image-feature extraction are working.
* Real VLF-aligned model tables are scaffolded for all-Italy and central Italy.
* Real PyTorch training should not start yet, because each real table has only one target class. The real deep patch Transformer wrapper records this blocker instead of training.
* Current real VLF-aligned label counts are 55 positive / 0 negative for all-Italy and 0 positive / 55 negative for central Italy.
* Self-supervised real VLF pretraining is available and is now the default model-development path until supervised labels have both classes.
* Historical seismic-only backfill for `2024-01-01` to `2026-07-07` produced 130 weekly training windows per scope.
* Backfilled all-Italy seismic windows are ready but heavily positive-skewed: train 95 positive / 9 negative, test 25 positive / 1 negative.
* Backfilled central-Italy seismic windows are more balanced: train 13 positive / 91 negative, test 6 positive / 20 negative.
* Current seismic-only temporal smoke scores are weak: all-Italy calibrated balanced accuracy `0.740000` on a one-negative test set; central Italy calibrated balanced accuracy `0.441667`.

Synthetic-model status:

* The current relabeled synthetic model table has 501 rows, with 67 positives and 434 negatives.
* CPU PyTorch tabular and GRU sequence models are implemented and compared.
* Current corrected-label temporal smoke scores remain weak: tabular PyTorch calibrated balanced accuracy `0.507576`; sequence GRU calibrated balanced accuracy `0.500000`.
* Current corrected-label seed holdouts are stronger but synthetic-only: tabular PyTorch ranges from `0.732602` to `0.784169`, while sequence GRU ranges from `0.720690` to `0.821105`.
* Older sequence sweeps, matched comparisons, repeated training-seed runs, and tiny patch Transformer checks were produced before the target relabeling and should be rerun before model selection.
* Corrected-label temporal split diagnostics still show a label/regime shift: `gt0` train positive rate `0.080000`, test positive rate `0.346535`; largest drift features are direct-avalanche active-topple and summary-topple aggregates.
* A post-burn-in regime-balanced explicit split has matched train/test class rates and gives `sequence_full` calibrated balanced accuracy `0.650000`; use this as an engineering diagnostic, not as forecasting evidence.
* The selected deeper patch Transformer pretrain now writes `deep_patch_transformer_synthetic.pt`; its latest synthetic calibrated scores are `0.737879` for piezo/VLF-only and `0.583333` for full sequence.
* The first self-supervised real VLF autoencoder smoke used 247 Cumiana VLF rows and 224 windows. Test masked reconstruction MSE was `0.835488`, better than the zero baseline `1.074356`.
* A second label-free real VLF anomaly layer now scores descriptor reconstruction and embedding novelty by window. The current 7-day smoke forecast artifact covers `2026-07-06T06:50:50Z` to `2026-07-13T06:50:50Z`, with demo probability `0.952514` and demo predicted event `1`; this is not trained on earthquake labels and is not a validated forecast.
* The tuned shape-profile synthetic-to-real embedding-domain diagnostic encoded 59,931 synthetic piezo/VLF windows through a descriptor autoencoder trained on real VLF windows. Synthetic centroid distance was `1.291640` and synthetic-to-real nearest mean distance was `1.846295`.
* The same diagnostic is still only a baseline, but it is stronger than the previous full-descriptor version: held-out real masked reconstruction MSE is `0.895188` versus a zero baseline of `0.960585`; synthetic masked reconstruction is `4.642818` versus a zero baseline of `4.721987`.
* A real-like synthetic inlier subset now marks the closest 25% of synthetic descriptor windows. It keeps 14,983 synthetic windows and reduces synthetic-to-real nearest mean distance to `1.162097`, with scale mean absolute delta `0.057490`.
* A synthetic-inlier transfer diagnostic now trains the masked descriptor autoencoder only on those 14,983 synthetic windows and evaluates on held-out real VLF descriptors. Held-out real masked reconstruction MSE is `0.688280` versus a zero baseline of `0.759011`, but the transfer embedding centroid distance remains high at `4.281796`.
* A mixed-domain alignment diagnostic now trains on real VLF plus 14,983 locally selected synthetic piezo/VLF windows with a CORAL embedding-alignment penalty. Held-out real masked reconstruction improves to `0.294475` versus a zero baseline of `0.588513`, and held-out embedding centroid distance improves to `1.033580`.
* Mixed-domain controls remain important: centroid-inlier selection scored `1.011474`, random synthetic selection `1.142438`, and capped full-synthetic selection `1.617345` on held-out centroid distance. This means alignment training is useful, but the local inlier criterion is not yet clearly superior to centroid selection.
* A first end-to-end trial weekly event-list forecast now combines current INGV history, VLF context, astronomy captures, and synthetic avalanche event artifacts. The `2026-07-08T00:00:00Z` run for the following week produced 25 capped `>M2` coordinate rows, with an uncapped expected count proxy of `33.411669`; this is a contract smoke test and not a validated prediction.
* A short piezo/VLF transform sweep added deterministic high-pass, burst, near-threshold, release-mix, and sensor-gain variants. The best transformed variant, `gain_burst`, improved short-run held-out embedding centroid distance to `1.757251` versus `1.841903` for the current signal, but worsened held-out masked reconstruction to `0.318687` versus `0.281600`.
* Refreshed missing-modality seed-42 checks give `0.632445` calibrated balanced accuracy for piezo/VLF-only and `0.722257` for direct-avalanche-only.
* Refreshed sequence modality diagnostics still rank direct-avalanche-only highest on grouped synthetic checks (`0.8359` calibrated balanced accuracy), so direct seismic-like and piezo/VLF-like channels should remain separate.
* A short diversity smoke run generated extra 128x128, 1000-step seeds `43` and `44` and refreshed aligned tensors with evaluations disabled; use larger runs before drawing model conclusions.
* Aligned synthetic targets now use true future look-ahead semantics: `target_horizon_rows=N` labels an input row from the sum of direct avalanche events in the next `N` complete rows, not from the current row or a single offset row.

## Shape Diagnostics

Real seismic vs synthetic seismic event traces:

* central-Italy seed `42` sparse-event distance: `1.4709`
* all-Italy seed `42` sparse-event distance: `1.3992`
* central-Italy real seismic is very sparse: nonzero ratio `0.0245`, PSD slope `-0.0380`
* all-Italy real seismic is less sparse: nonzero ratio `0.1756`, PSD slope `-0.0272`
* current synthetic seismic events remain too dense: nonzero ratio `0.4275`, PSD slope `0.0689`
* raw synthetic avalanche signal is effectively continuous: nonzero ratio `0.9991`

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
* refined sparse tuning adds `max_events` and a stable `shape_score` independent of candidate-grid size
* refined 10000-step seed `42` central-Italy sparse profile: `q=0.99`, window `480`, max events `3`
* refined 20000-step seeds `40`-`42` consistently prefer: `q=0.999`, window `240`, max events `5`
* refined 20000-step sparse profile event shape scores: seed `40` `0.193665`, seed `41` `0.119737`, seed `42` `0.144586`

## Interpretation

Sparse local-peak extraction improved the direct synthetic seismic event trace. The extended 2024-2026 INGV comparison showed the previous synthetic seismic event list was still too dense, especially against central Italy. The refined sparse profile improves 10000-step seed `42` central-Italy distance from `1.4709` to `1.0218`, with nonzero ratio `0.0259` versus real `0.0245`. Keep it as a sparse seismic profile until downstream target balance is re-evaluated.

Across the full-size seed check, seed `42` is still the best current default on direct seismic event-shape distance and sparsity, but all tested seeds remain much denser than real seismic events. The next tuning target is therefore direct avalanche event extraction thresholds and burst spacing, not map projection.

The old extraction tuning pass supported quantile `0.99` and local-max window `30` as a usable model-training default. The refined sparse profile uses far fewer events (`q=0.999`, window `240`, max events `5` on 20000-step runs) and better matches central-Italy sparsity, but it may make synthetic model targets too sparse if promoted without redesigning the target windows.

The target-window redesign is now implemented. On the refined sparse seed `40`-`42`, `20000`-step profile, positive labels scale with the look-ahead horizon as expected: horizon `1` has `3/501` positives, horizon `3` has `9/495`, horizon `6` has `18/486`, horizon `12` has `36/468`, and horizon `24` has `72/432`. Temporal splits still have zero positive test rows because the sparse synthetic events occur too early in each run, so the sparse profile should not become the default until event timing is less clustered or substantially longer synthetic runs are available.

The piezo/VLF image rendering is now much closer to Cumiana image statistics for brightness, high-intensity coverage, and vertical streaks. The underlying piezo time series is still too smooth in time, with very high autocorrelation, so later tuning should focus on signal dynamics rather than adding display artifacts.

The per-sensor scan shows that summing all piezo sensors over-smooths the VLF-like channel. Single sensors are more plausible. Raw burst-run counts were misleading because the image-derived VLF trace has many more samples than the simulation trace, so comparisons now use burst-run rate.

The current tuned default uses thresholded accumulated-charge release plus a local receiver footprint. It improves lag-1 autocorrelation and the corrected overall shape score without artificial spikes. The seed `40`-`42` validation is consistent enough to keep these as current simulation defaults, but the best receiver is seed-dependent, so model data preparation should preserve sensor identity and support sensor selection or pooling.

Piezo receiver locality is now separated from the direct avalanche signal range. Tuning the VLF-like piezo channel should not silently change the seismic-like avalanche channel.

After that separation and the target relabeling fix, seed `40`-`42` simulation CSVs, aligned synthetic model rows, tensors, temporal diagnostics, tabular PyTorch reports, sequence GRU reports, and the tabular-vs-sequence comparison were refreshed. The chronological holdout remains weak, while leave-one-seed-out checks remain stronger and should be treated as synthetic transfer diagnostics only.

Longer synthetic run check:

* regenerated seed `40`, `41`, and `42` at `20000` steps under current defaults
* sparse event counts: seed `40` has `130`, seed `41` has `129`, seed `42` has `135`
* combined `gt0` aligned table after future look-ahead relabeling: `501` rows, `67` positives, `434` negatives
* combined `gt1` aligned table after future look-ahead relabeling: `501` rows, `2` positives, `499` negatives
* `gt0` chronological best calibrated balanced accuracy: `0.507576`
* `gt0` leave-one-seed-out best calibrated balanced accuracy range: `0.591536` to `0.826389`
* `gt1` remains mostly a sparsity check

Chronological split diagnostics:

* diagnostic outputs:
  * `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.temporal_diagnostics.json`
  * `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.temporal_diagnostics.csv`
  * `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows_gt1.temporal_diagnostics.json`
  * `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows_gt1.temporal_diagnostics.csv`
* `gt0` train positive rate is `0.080000`; test positive rate is `0.346535`
* `gt1` train positive rate is `0.000000`; test positive rate is `0.019802`
* later test windows have much higher avalanche/topple intensity: `synthetic_direct_avalanche_active_topple_cell_count_*` and `synthetic_summary_topple_count_*` drift upward by more than one training standard deviation
* several direct avalanche and summary features still change target-correlation strength between train and test in the `gt0` split

The longer run increases target support but does not improve chronological generalization. It should be used to stress-test the model interface and synthetic-transfer workflow, not as evidence of predictive value.

The weak `20000` chronological holdout appears to be mostly a non-stationary split problem. The simulation trajectory changes regime over time: later windows are taller and more avalanche-active, and the target rate changes substantially. Time-split evaluation is still the right conservative check, but the current synthetic generator needs burn-in handling, regime-balanced splitting, or longer/more varied runs before model metrics are meaningful.

The direct avalanche signal should remain separate from the piezo/VLF channel. Cross-modality distances can be useful sanity checks, but tuning should compare real seismic primarily with direct avalanche outputs, and real VLF primarily with piezo-derived outputs.

## Model Scaffold Update

Current CPU PyTorch tabular and sequence reports can now be compared directly with `compare-model-runs.sh`. The sequence path also has a bounded sweep script, a missing-modality smoke script, and a real Cumiana VLF image sequence materializer so synthetic piezo/VLF and real VLF inputs keep the same model-facing shape.

Corrected-label smoke outputs:

* tabular-vs-sequence comparison: `data/derived/models/mountain_256x256_seeds40-42_20000.tabular_vs_sequence_model_comparison.json`
* best calibrated row in that comparison: `0.826389`, `seismic_vlf_unified`, held-out `seed41`
* tabular PyTorch temporal row: calibrated balanced accuracy `0.507576`
* tabular PyTorch seed holdouts: calibrated balanced accuracy `0.784169`, `0.762832`, and `0.732602`
* sequence GRU temporal row: calibrated balanced accuracy `0.500000`
* sequence GRU seed holdouts: calibrated balanced accuracy `0.720690`, `0.821105`, and `0.768339`
* real VLF image sequence manifest: `data/derived/models/cumiana_vlf_image_sequence/manifest.json`, with `247` time steps and `25` channels
* prospective VLF-aligned labels currently have `23` all-Italy positives and `0` negatives, while central Italy has `0` positives and `23` negatives, so real VLF-aligned training still has insufficient class variation

Pre-relabel sweep outputs that need rerunning before model selection:

* tiny sequence sweep smoke: `data/derived/models/sequence_sweep_smoke/sequence_sweep_comparison.json`; best calibrated row `0.709624`, `sequence_full`, held-out `seed41`
* missing-modality smoke: `data/derived/models/missing_modality/missing_modality_seed42_summary.json`; no-piezo direct avalanche scored higher than piezo-only in this short run
* full sequence sweep: `data/derived/models/sequence_sweep/sequence_sweep_comparison.json`, `24` reports; best calibrated row `0.766942`, `sequence_direct_avalanche_only`, `lookback=60`, `hidden=24`, held-out `seed42`
* combined family comparison: `data/derived/models/model_family_comparison.json`, `37` rows; pre-relabel best calibrated row `0.772558`, `sequence_piezo_vlf_only`, held-out `seed42`
* sequence modality diagnostic: `data/derived/models/sequence_modality_diagnostic.json`, `112` evaluation rows; best default sequence row uses `20` epochs and piezo/VLF-only, while best sweep row uses `10` epochs and direct avalanche-only
* matched 20-epoch sequence comparison: `data/derived/models/sequence_sweep_20epoch/default_vs_matched_sequence_diagnostic.json`, `64` evaluation rows; pre-relabel best row `sequence_piezo_vlf_only`, `lookback=60`, `hidden=24`, held-out `seed42`, calibrated balanced accuracy `0.772558`
* repeated training-seed comparison: `data/derived/models/sequence_training_seed_repeat/sequence_training_seed_selection.json`; pre-relabel best single row `sequence_piezo_vlf_only` at `0.772558`, with `sequence_full` best on mean group score (`0.741342`) and worst-held-out-seed score (`0.712754`)
* tiny patch Transformer scaffold: `data/derived/models/tiny_patch_transformer/tiny_patch_transformer_model_run_summary.json`; pre-relabel best calibrated row `0.637500`, `sequence_piezo_vlf_only`, explicit regime-balanced split
* real model-input scaffold: `data/derived/models/all_italy.real_vlf_aligned_windows.csv` and `data/derived/models/central_italy.real_vlf_aligned_windows.csv`; both have `247` rows and `23` labeled rows but still lack class variation

Sequence diagnostic interpretation: do not change the default GRU lookback from `60` on current evidence. The corrected-label temporal sequence row remains at balanced accuracy `0.5`; older sweep and missing-modality reports need rerunning before choosing between direct avalanche, piezo/VLF, and full sequence inputs.

Post-burn-in regime interpretation: `sequence_full` does not yet show robust performance once the first 20 percent of each synthetic seed is removed and holdouts are made by seed/regime block. This supports the earlier concern that the current synthetic generator and split design still contain regime effects that should be understood before larger model runs.

Compact model comparison:

* artifact: `data/derived/models/real_synthetic_compact_comparison.json`
* CSV view: `data/derived/models/real_synthetic_compact_comparison.csv`
* central-Italy historical seismic-only temporal baseline: calibrated balanced accuracy `0.441667`
* corrected-label synthetic temporal sequence run: calibrated balanced accuracy `0.500000`
* corrected-label best synthetic seed-holdout row: `seismic_vlf_unified`, held-out `seed41`, calibrated balanced accuracy `0.826389`
* post-burn-in `sequence_full` regime holdouts: mean calibrated balanced accuracy `0.508413`
* post-burn-in `sequence_full` regime-balanced split: calibrated balanced accuracy `0.650000`
* pre-relabel tiny patch Transformer regime-balanced split: `sequence_piezo_vlf_only`, calibrated balanced accuracy `0.637500`

## Limitations

The real VLF data is image-derived, not raw waveform data.

The current historical seismic sample covers January 1, 2024 through July 7, 2026. It is large enough for smoke baselines and shape diagnostics, but still not enough for robust predictive claims.

Simulation step time is an assumed mapping. Frequency-domain comparisons are therefore shape diagnostics, not physical frequency validation.

The current direct avalanche signal input uses `*.avalanche_signal.csv`; model smoke runs currently use the 20000-step seed `40`-`42` synthetic tables.
