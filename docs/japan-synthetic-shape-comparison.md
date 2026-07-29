# Japan And Synthetic Shape Comparison

## Scope

The diagnostic compares the latest 11 Moshiri CDF feature captures and the Japan normalized seismic catalog with the seed-40, 20,000-step avalanche episode. Outputs are:

* `data/derived/reports/japan_synthetic_signal_shape_series.csv`
* `data/derived/reports/japan_synthetic_signal_shape_pairs.csv`

The Japan VLF trace is the mean of the 16 native `ch1`/`ch2` logarithmic power bands. It is a compact spectrogram-feature projection, not a raw radio waveform. Captures are concatenated for shape statistics; gaps are not physical continuous time. Event-energy series use one-hour bins and `10^magnitude`, so their absolute power is not comparable with sensor amplitudes.

## Results

| Series | CV | Nonzero | Burst ratio | Lag-1 autocorrelation | Skew | Excess kurtosis | PSD slope | PSD low-band ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Japan VLF log power | 0.074 | 1.000 | 0.000 | 0.599 | -1.367 | 11.223 | -0.267 | 0.771 |
| Japan seismic event energy | 33.766 | 0.084 | 0.040 | 0.013 | 73.448 | 5974.232 | -0.023 | 0.351 |
| Synthetic seismic event energy | 1.297 | 0.401 | 0.051 | 0.202 | 0.883 | -0.187 | -0.212 | 0.437 |
| Synthetic piezo/VLF signal | 0.219 | 1.000 | 0.050 | 0.841 | 2.490 | 7.824 | 0.248 | 0.361 |
| Synthetic direct avalanche signal | 0.165 | 1.000 | 0.050 | 0.316 | -0.372 | 2.006 | 0.299 | 0.317 |

## Interpretation

1. The synthetic seismic event process is too dense and too regular relative to Japan. Its nonzero fraction is about five times higher, its skew and kurtosis are dramatically lower, and its lag correlation is higher. This supports reducing event rate and increasing episodic clustering before using the catalog for transfer.
2. The Japan VLF projection is low-frequency dominated: 77.1% of spectral power is in the lowest third and its PSD slope is negative. The current piezo signal has a positive PSD slope and only 36.1% low-band power. Its `0.841` lag correlation also indicates smooth persistence rather than the intermittent, heavy-tailed radio-like structure suggested by the Japan feature trace.
3. The Japan VLF distribution is bounded by the log-power floor and therefore should not be matched by raw mean, coefficient of variation, or amplitude. Shape matching should use per-capture standardized anomalies, band contrasts, burst duration, and PSD ratios.
4. The Japan and synthetic seismic spectra are not yet a strong physical comparison because their observation durations and event counts differ. The event-process mismatch is nevertheless robust in sparsity and tail statistics.

## Simulation Priorities

1. Make the synthetic direct event extractor sparser and more clustered using a train-only rate calibration, then validate on held-out seeds. Do not simply add random noise or duplicate events.
2. Add a slow envelope to the piezo sensor derived from accumulated causal stress/charge, with intermittent release modulation. The envelope should increase low-frequency power while preserving avalanche-derived timing.
3. Replace global amplitude matching with per-capture robust normalization and compare band-relative VLF features. Preserve raw outputs and keep any transform as a declared observation model.
4. Re-run this report over several synthetic seeds and more Japan captures. Treat a single seed-40 comparison as a diagnostic, not a simulator-selection result.

The comparison does not demonstrate that Japan VLF predicts earthquakes. It identifies measurable signal-shape targets for improving the synthetic observation model and avalanche event calibration.

## Observation-model experiment

The first implementation is `--slow-envelope-decay` plus `--slow-envelope-mix` in `transform-piezo-signal`. It applies causal amplitude modulation to the existing avalanche-derived signal; it does not add independent events, timestamps, or noise. Use `scripts/evaluate-piezo-japan-shape-variants.sh` to compare weak, strong, and longer-memory settings. A setting is useful only if it improves low-frequency and tail statistics without making the signal a featureless trend; it must then be checked across seeds.

The first four-variant run did not meet that gate. The strong setting moved excess kurtosis from `21.04` to `43.94`, but PSD slope changed only from `0.517` to `0.501` and low-band power from `0.169` to `0.170`, versus Japan values of `-0.267` and `0.771`. The slow-envelope stage is therefore retained as an opt-in observation-model experiment, not a default.

The separate event-extraction probe is in `data/derived/reports/japan-avalanche-event-tuning-reduced.csv`. Its best single-episode diagnostic uses the `0.975` signal quantile, a 120-step local-maximum window, and a 25-event cap. It matches the Japan nonzero fraction (`0.08446` versus `0.08430`) and event PSD slope (`0.0013` versus `-0.0230`), but its burst-run count is much lower (`7` versus `461`). This is evidence that rate calibration is useful, not evidence that a global cap is physically correct.

The fixed-policy seven-seed check is in `data/derived/reports/japan-avalanche-policy-seeds/summary.csv`. The cap is reached in every episode, while nonzero fraction ranges from `0.0776` to `0.0883`. Burst-run counts remain only `4` to `16`, and lag-1 autocorrelation remains about `-0.09` across seeds. The extraction policy is therefore stable enough for a control, but the underlying avalanche process still lacks the clustered temporal structure needed for a stronger synthetic analogue.

The first simulation-level test used `OUTPUT_TAG=.clustered`, `SOURCE_ACTIVITY_DECAY=0.6`, `SOURCE_ACTIVITY_BOOST=0.8`, and deposition probability `0.3`. The 5,000-step episode remained bounded near mean height `127.7`, but its extracted event trace still had only four burst runs and PSD slope `0.898`. Per-source persistence alone is not sufficient; the next regime must correlate loading intensity across the fixed source set.

The shared-regime follow-up used `SOURCE_REGIME_DECAY=0.2`, `SOURCE_REGIME_BOOST=0.8`, and `TARGET_FILL_REGIME_FLOOR=0.25`. It prevented early target refill from masking the loading state, but the 5,000-step event trace still had four burst runs, lag-1 correlation `-0.320`, and PSD slope `0.638`. Neither tested loading regime should be promoted. The next improvement should focus on how avalanche activity is aggregated into events, or on a longer-lived stress-release mechanism, rather than adding more activation multipliers.

The activity table explains why. The baseline avalanche activity is nonzero on `99.96%` of steps, and the shared-regime screening run is nonzero on `99.98%`. The current extractor therefore ranks peaks within an almost continuously active process; its local-maximum window and global event cap control the catalog more strongly than distinct quiet-to-active episodes. A future extractor should use a causal baseline-subtracted burst score or explicit episode segmentation, with calibration performed on training episodes only.

## Causal burst extractor

`scripts/probe-avalanche-burst-extractor.sh` evaluates causal baseline subtraction followed by one peak per above-threshold episode. The best single-episode diagnostic is baseline decay `0.99` with a 120-step gap and no event cap: nonzero ratio `0.0697`, PSD slope `0.0159`, and excess kurtosis `10.62`. These are closer to Japan’s `0.0843`, `-0.0230`, and `5974.23` than the earlier capped extractor, although burst-run count is still only `16` versus `461` and lag-1 correlation is `-0.070` versus `0.013`. This is a promising extraction control, not evidence of physical realism or predictive value.

The seven-seed validation is in `data/derived/reports/avalanche-burst-seeds/summary.csv`. With the same decay and gap, nonzero ratio ranges `0.050–0.090`, excess kurtosis `6.95–14.99`, and burst-run count `12–17`. PSD slopes range `0.016–0.167`; seeds `40` and `4600` are closest to the Japan slope, but the sign and magnitude are not stable enough for promotion. The current script calibrates the threshold independently per episode for diagnosis; a training artifact must estimate that threshold on training episodes and apply it unchanged to held-out episodes.

The leakage-safe train/test check is in `data/derived/reports/avalanche-burst-train-test/summary.csv`. A single threshold of `750.965` learned from seeds `40,41,42,4300` gives held-out nonzero ratios `0.101–0.170`, against the Japan reference `0.084`. This is a clear regime-scale failure: the score has useful shape information, but its raw amplitude is not invariant across seeds. The next calibration should normalize each episode against a training-derived robust scale or use regime-conditioned calibration; the held-out threshold must remain untouched.

The first causal relative-baseline control uses the same train/test split and a threshold of `0.221864`. It improves seed-40 rate matching, but held-out nonzero ratios are `0.145–0.176`, worse than the raw-score control. Relative baseline normalization is therefore rejected as a sufficient fix; the episode-generating process remains regime dependent.

The bounded source-stress reservoir was then tested for 5,000 steps. It was numerically safe, but the causal burst extractor selected only one event. This rejects the tested reservoir parameters as a direct-event improvement; its release mechanism is too small or too persistent relative to the continuously active mountain state.

A per-source cooldown control (`SOURCE_STRESS_RELEASE_COOLDOWN_STEPS=120`) was also tested for 5,000 steps. It remained safety-release free but still produced one extracted direct event. The cooldown separates releases locally, not globally, so it is retained only as a simulator control.
