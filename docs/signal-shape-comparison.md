# Signal Shape Comparison

Purpose: compare real seismic/VLF observations with simulation-derived signals before training larger models.

Run:

```sh
./scripts/compare-signal-shapes.sh
```

Outputs:

* `*.signal_shape_series.csv` - per-series time-domain and PSD metrics
* `*.signal_shape_pairs.csv` - pairwise normalized distances and metric deltas

Series currently supported:

* normalized INGV-like event CSVs binned into seismic energy traces
* Cumiana VLF spectrogram JPGs reduced to column-intensity traces
* `*.piezo.csv` reduced to the synthetic VLF analogue trace
* `*.avalanche_signal.csv` reduced to the direct seismic analogue trace

Metrics include distribution shape, burstiness, lag-1 autocorrelation, PSD slope, spectral centroid, rolloff, and low/mid/high power ratios.

Pairwise `normalized_distance` uses dimensionless shape and PSD-ratio metrics. Absolute amplitude, sample count, and duration are still reported as deltas but are not included in that distance.

See [Simulation Time Scale](simulation-time-scale.md) before interpreting PSD metrics.

Interpretation rules:

* compare seismic event traces against direct avalanche traces
* compare VLF image traces against piezo traces
* do not compare raw physical frequencies across modalities without a declared time-scale mapping
* treat distances as tuning diagnostics, not validation of predictive power
