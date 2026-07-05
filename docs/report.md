# Analysis Report

Date: 2026-07-03

## Scope

This report summarizes the first statistical comparison between real Italy-scoped seismic/VLF data and signals derived from the avalanche simulation.

The comparison is diagnostic only. It does not demonstrate earthquake prediction capability.

## Inputs

Real seismic:

* `data/derived/ingv/events_italy_2026-06-01_2026-06-29.normalized.csv`
* 151 normalized INGV event rows

Real VLF:

* 115 Cumiana `last_E_VLF` spectrogram captures
* capture range represented in the current data: 2026-06-29 to 2026-07-03

Synthetic:

* `data/derived/sim/mountain_256x256_seed42_10000.piezo.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.avalanche_events.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.avalanche_signal.csv`

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

## Key Results

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

The longer run increases target support but does not improve chronological generalization. It should be used to stress-test the model interface and synthetic-transfer workflow, not as evidence of predictive value.

The direct avalanche signal should remain separate from the piezo/VLF channel. Cross-modality distances can be useful sanity checks, but tuning should compare real seismic primarily with direct avalanche outputs, and real VLF primarily with piezo-derived outputs.

## Limitations

The real VLF data is image-derived, not raw waveform data.

The current real seismic sample covers only June 2026 and is too short for robust statistical claims.

Simulation step time is an assumed mapping. Frequency-domain comparisons are therefore shape diagnostics, not physical frequency validation.

The current 10000-step direct avalanche signal input now uses `*.avalanche_signal.csv`.
