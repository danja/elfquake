# Next Actions

## Simulation

1. Re-run the 10000-step default simulation so `*.avalanche_signal.csv` replaces the legacy `*.piezo_avalanche.csv` input.
2. Compare multiple simulation seeds and parameter sets with `compare-simulation-grid.sh` before accepting tuning changes.
3. Tune direct avalanche peak thresholds against real INGV sparsity and burst metrics.
4. Tune piezo/VLF display scaling against Cumiana intensity and vertical-streak metrics.
5. Backfill more INGV event windows and keep accumulating Cumiana VLF captures before model training claims.

## General

1. Reinstall/reload the updated prospective systemd unit if timer-managed image features and summaries are desired.
2. Keep the VLF capture and prospective timers running until the first target windows mature.
3. On or after `2026-07-06T09:57:24Z`, refresh INGV events through `2026-07-07` and label the first prospective rows.
4. Add chunked sandpile snapshot storage only if larger pretraining runs outgrow `.npy` sanity snapshots.
5. Add slope/erosion smoothing to mountain-mode synthetic terrain if ridgeline-like visuals are needed.
6. Backfill enough historical INGV windows to get both positive and negative target classes by region.
7. Continue with prospective-only VLF evaluation unless a separate historical Cumiana archive is obtained.
8. Replace the schematic event-map basemap with a real cartographic basemap if geospatial dependencies are installed.
9. Add avalanche centroid or rupture-mask outputs so synthetic event maps are not limited to fixed sensor proxy locations.
10. Define regular-cadence multimodal tensors with missing-data masks before implementing Transformer candidates.
11. Use the piezo/Cumiana comparison report to tune only the piezo VLF mapping derived from `*.piezo.csv`.
12. Shape direct avalanche signal events for INGV-like seismic event experiments without using the piezo/VLF path.
13. Use the signal-shape comparison report to tune simulation parameters separately for VLF-like and seismic-like outputs.

## Completed

* Add sparse local-peak extraction for direct avalanche-derived seismic events.
* Add VLF display scaling controls derived from the piezo signal.
* Add `compare-simulation-grid.sh` for multi-seed simulation comparison.
* Add a declared simulation-to-real time-scale note before PSD interpretation.
