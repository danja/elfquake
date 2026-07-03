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

* 112 Cumiana `last_E_VLF` spectrogram captures
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

* shape distance after sparse direct avalanche peak extraction: `1.5256`
* real seismic is sparse: nonzero ratio `0.1583`
* synthetic event trace is less dense than before: nonzero ratio `0.7652`
* real seismic PSD slope: `-0.0670`
* synthetic event PSD slope: `-0.1678`

Real VLF image columns vs synthetic piezo/VLF signal:

* shape distance: `1.4674`
* real VLF PSD slope: `-0.3556`
* synthetic piezo PSD slope: `-0.2705`
* real VLF lag-1 autocorrelation: `0.5271`
* synthetic piezo lag-1 autocorrelation: `0.9658`

VLF image-level comparison:

* nearest Cumiana image: `data/raw/vlf/cumiana/captures/2026-06-30/last_E_VLF_2026-06-30T23-15-00Z.jpg`
* nearest normalized distance after render tuning: `13.6424`
* simulated intensity mean: `0.5987`
* real mean intensity: `0.4736`
* simulated high-intensity ratio: `0.1695`
* real high-intensity ratio: `0.1745`
* simulated vertical streak count: `115`
* real mean vertical streak count: `119.3750`

Multi-seed smoke check:

* `compare-simulation-grid.sh` ran for seeds `40` and `41` on `32 x 32`, `200` step runs
* sparse direct avalanche event extraction produced `7` and `4` event rows
* the smoke reports confirm the comparison machinery can be reused across seeds, but the small runs should not be used for tuning conclusions

## Interpretation

Sparse local-peak extraction improved the direct synthetic seismic event trace. It is still denser than the real INGV trace, but the PSD slope and overall distance are much closer than the previous all-nonzero event extraction.

The piezo/VLF image rendering is now much closer to Cumiana image statistics for brightness, high-intensity coverage, and vertical streaks. The underlying piezo time series is still too smooth in time, with very high autocorrelation, so later tuning should focus on signal dynamics rather than adding display artifacts.

The direct avalanche signal should remain separate from the piezo/VLF channel. Cross-modality distances can be useful sanity checks, but tuning should compare real seismic primarily with direct avalanche outputs, and real VLF primarily with piezo-derived outputs.

## Limitations

The real VLF data is image-derived, not raw waveform data.

The current real seismic sample covers only June 2026 and is too short for robust statistical claims.

Simulation step time is an assumed mapping. Frequency-domain comparisons are therefore shape diagnostics, not physical frequency validation.

The current 10000-step direct avalanche signal input now uses `*.avalanche_signal.csv`.
