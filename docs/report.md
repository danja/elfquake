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

Output files:

* `data/derived/sim/mountain_256x256_seed42_10000.signal_shape_series.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.signal_shape_pairs.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.piezo_vlf_comparison.csv`

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

Full-size multi-seed check:

* `compare-simulation-grid.sh` ran for full-size `256 x 256`, `10000` step runs on seeds `40`, `41`, and `42`
* metrics-only runs used `RUN_HEATMAPS=0`, `RUN_VIDEO=0`, and `RUN_AUDIO=0`
* summary CSV: `data/derived/sim/full_size_seed_comparison.csv`

| seed | events | seismic distance | event nonzero | event PSD slope | piezo/VLF distance | piezo lag-1 | nearest VLF image distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 189 | 1.5775 | 0.8030 | 0.2524 | 1.4041 | 0.9581 | 13.2128 |
| 41 | 191 | 1.6198 | 0.8106 | -0.4042 | 1.4663 | 0.9666 | 13.4896 |
| 42 | 172 | 1.5258 | 0.7652 | -0.1678 | 1.4620 | 0.9658 | 13.6492 |

## Interpretation

Sparse local-peak extraction improved the direct synthetic seismic event trace. It is still denser than the real INGV trace, but the PSD slope and overall distance are much closer than the previous all-nonzero event extraction.

Across the full-size seed check, seed `42` is still the best current default on direct seismic event-shape distance and sparsity, but all tested seeds remain much denser than real seismic events. The next tuning target is therefore direct avalanche event extraction thresholds and burst spacing, not map projection.

The piezo/VLF image rendering is now much closer to Cumiana image statistics for brightness, high-intensity coverage, and vertical streaks. The underlying piezo time series is still too smooth in time, with very high autocorrelation, so later tuning should focus on signal dynamics rather than adding display artifacts.

The direct avalanche signal should remain separate from the piezo/VLF channel. Cross-modality distances can be useful sanity checks, but tuning should compare real seismic primarily with direct avalanche outputs, and real VLF primarily with piezo-derived outputs.

## Limitations

The real VLF data is image-derived, not raw waveform data.

The current real seismic sample covers only June 2026 and is too short for robust statistical claims.

Simulation step time is an assumed mapping. Frequency-domain comparisons are therefore shape diagnostics, not physical frequency validation.

The current 10000-step direct avalanche signal input now uses `*.avalanche_signal.csv`.
