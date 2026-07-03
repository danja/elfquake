# Next Actions

## Simulation

1. Tune direct avalanche peak thresholds against real INGV sparsity and burst metrics.
2. Tune piezo signal dynamics so lag-1 autocorrelation moves closer to real VLF image-column traces.
3. Backfill more INGV event windows and keep accumulating Cumiana VLF captures before model training claims.

## General

1. Reinstall/reload the updated prospective systemd unit if timer-managed image features and summaries are desired.
2. Keep the VLF capture and prospective timers running until the first target windows mature.
3. On or after `2026-07-06T09:57:24Z`, refresh INGV events through `2026-07-07` and label the first prospective rows.
4. Add chunked sandpile snapshot storage only if larger pretraining runs outgrow `.npy` sanity snapshots.
5. Add slope/erosion smoothing to mountain-mode synthetic terrain if ridgeline-like visuals are needed.
6. Backfill enough historical INGV windows to get both positive and negative target classes by region.
7. Continue with prospective-only VLF evaluation unless a separate historical Cumiana archive is obtained.
8. Add full rupture-mask outputs if synthetic event maps need spatial extent rather than centroid locations.
9. Generate a longer or multi-seed synthetic aligned dataset to reduce time-split distribution drift.
10. Use the piezo/Cumiana comparison report to tune only the piezo VLF mapping derived from `*.piezo.csv`.
11. Shape direct avalanche signal events for INGV-like seismic event experiments without using the piezo/VLF path.
12. Use the signal-shape comparison report to tune simulation parameters separately for VLF-like and seismic-like outputs.

## Completed

* Add naive baseline and threshold-calibrated metrics to temporal holdout reports.
* Generate an hourly synthetic aligned dataset with enough rows for a non-trivial chronological holdout check.
* Add a time-ordered train/test smoke evaluator for aligned synthetic and real labeled rows.
* Align seismic, VLF, astronomy, and simulation features onto regular-cadence model rows for current synthetic and real data paths.
* Add simulation time-scale metadata so step-indexed sequence tensors can be aligned to UTC windows.
* Add explicit VLF capture timestamps to VLF tensor indexes while preserving source-file provenance.
* Add an alignment manifest that links window tensors, sequence tensors, source time ranges, and ablation groups for one model run.
* Add event-window adapter for real and synthetic seismic event lists.
* Add sequence materializer for `time x sensor x channel` simulation and VLF-like signals.
* Add model-interface shape audit for event lists, image feature tables, sensor time series, and summary series.
* Add sparse local-peak extraction for direct avalanche-derived seismic events.
* Add backend-neutral tensor materialization from tensor specs with values, masks, and row index files.
* Add modular model-candidate registry and tensor-spec scaffold for future Transformer work.
* Compare full-size `256 x 256`, `10000` step seeds `40`, `41`, and `42` without heatmap/video overhead.
* Add a Natural Earth Italy line basemap for avalanche-derived event-map demos.
* Add direct avalanche activity centroids for synthetic seismic event locations.
* Scale avalanche-derived demo event locations over an Apennine-style Italy profile with magnitude-sized map markers.
* Add VLF display scaling controls derived from the piezo signal.
* Add `compare-simulation-grid.sh` for multi-seed simulation comparison.
* Add a declared simulation-to-real time-scale note before PSD interpretation.
* Re-run the 10000-step default simulation so `*.avalanche_signal.csv` replaces the legacy `*.piezo_avalanche.csv` input.
