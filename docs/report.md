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

* 103 Cumiana `last_E_VLF` spectrogram captures
* capture range represented in the current data: 2026-06-29 to 2026-07-03

Synthetic:

* `data/derived/sim/mountain_256x256_seed42_10000.piezo.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.avalanche_events.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.piezo_avalanche.csv` as the legacy direct avalanche signal file

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

* shape distance: `2.1199`
* real seismic is sparse: nonzero ratio `0.1583`
* synthetic event trace is dense: nonzero ratio `1.0000`
* real seismic PSD slope: `-0.0670`
* synthetic event PSD slope: `-1.1973`

Real VLF image columns vs synthetic piezo/VLF signal:

* shape distance: `1.3937`
* real VLF PSD slope: `-0.3487`
* synthetic piezo PSD slope: `-0.2705`
* real VLF lag-1 autocorrelation: `0.5033`
* synthetic piezo lag-1 autocorrelation: `0.9658`

VLF image-level comparison:

* nearest Cumiana image: `data/raw/vlf/cumiana/captures/2026-07-02/last_E_VLF_2026-07-02T18-00-00Z.jpg`
* nearest normalized distance: `65.4859`
* simulated intensity mean: `0.1231`
* real mean intensity: `0.4858`
* simulated high-intensity ratio: `0.0002`
* real high-intensity ratio: `0.1833`
* simulated vertical streak count: `0`
* real mean vertical streak count: `122.1262`

## Interpretation

The direct synthetic seismic event output is not yet event-shaped enough. It produces activity in every hourly bin, while the real INGV event trace is mostly empty with occasional bursts. The synthetic event pipeline should therefore raise event thresholds or build events from stronger avalanche clusters rather than treating nearly every simulated interval as event-like.

The piezo/VLF time-series PSD slope is closer to the real VLF column-intensity slope than the image-level visual comparison suggests. However, the synthetic piezo trace is much too smooth in time, with very high autocorrelation, and the rendered VLF image remains too dim and lacks vertical streaks. The issue is therefore partly signal dynamics and partly image mapping.

The direct avalanche signal should remain separate from the piezo/VLF channel. Cross-modality distances can be useful sanity checks, but tuning should compare real seismic primarily with direct avalanche outputs, and real VLF primarily with piezo-derived outputs.

## Limitations

The real VLF data is image-derived, not raw waveform data.

The current real seismic sample covers only June 2026 and is too short for robust statistical claims.

Simulation step time is an assumed mapping. Frequency-domain comparisons are therefore shape diagnostics, not physical frequency validation.

The current direct avalanche signal input uses a legacy filename, `*.piezo_avalanche.csv`; future simulation runs should use `*.avalanche_signal.csv`.


