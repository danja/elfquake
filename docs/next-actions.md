# Next Actions

## Simulation

1. Backfill more INGV event windows and keep accumulating Cumiana VLF captures before model training claims.
2. Compare leave-one-seed-out synthetic results against longer-run synthetic results before treating the generator as stable.
3. Add longer synthetic aligned datasets using the `0.99/30` event extraction defaults.
4. Add a small grouped-sensor piezo scan only if single-receiver traces prove too local after multi-seed validation.
5. Add a reusable script for refreshing synthetic aligned model artifacts if this pipeline is repeated often.

## General

1. Reinstall/reload the updated prospective systemd unit if timer-managed image features and summaries are desired.
2. Keep the VLF capture and prospective timers running until the first target windows mature.
3. On or after `2026-07-06T09:57:24Z`, refresh INGV events through `2026-07-07` and label the first prospective rows.
4. Add chunked sandpile snapshot storage only if larger pretraining runs outgrow `.npy` sanity snapshots.
5. Add slope/erosion smoothing to mountain-mode synthetic terrain if ridgeline-like visuals are needed.
6. Backfill enough historical INGV windows to get both positive and negative target classes by region.
7. Continue with prospective-only VLF evaluation unless a separate historical Cumiana archive is obtained.
8. Add full rupture-mask outputs if synthetic event maps need spatial extent rather than centroid locations.
9. Generate a longer synthetic aligned dataset to reduce time-split distribution drift.
10. Use the piezo/Cumiana comparison report to tune only the piezo VLF mapping derived from `*.piezo.csv`.
11. Shape direct avalanche signal events for INGV-like seismic event experiments without using the piezo/VLF path.
12. Use the signal-shape comparison report to tune simulation parameters separately for VLF-like and seismic-like outputs.
13. Add a small Markdown or CSV view over the compact model-run summary only if JSON becomes awkward to inspect.

## Completed

* Add naive baseline and threshold-calibrated metrics to temporal holdout reports.
* Add a reproducible direct avalanche event-extraction tuning helper and run it on seeds `40`, `41`, and `42`.
* Validate direct avalanche event quantile `0.99` on a longer seed `42`, `20000` step run and update pipeline defaults.
* Validate direct avalanche local-max windows on longer seed `40`, `41`, and `42` runs and update the window default to `30`.
* Rebuild current default avalanche event lists with `0.99/30` and refresh downstream synthetic model artifacts.
* Add a compact model-target note that the current smoke target is `gt0`, not `gt1`, after sparse event extraction.
* Switch chronological model smoke evaluation to an `80/20` train/test split.
* Add leave-one-seed-out group holdout evaluation for combined synthetic aligned rows.
* Add a compact model-run summary artifact that compares chronological and group-holdout reports side by side.
* Generate and materialize a combined seed `40`-`42` hourly synthetic aligned `gt1` dataset.
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
* Add a per-sensor piezo/VLF shape scanner and wrapper script; pre-tuning best sensor was `9`, tuned default uses sensor `5`.
* Add an optional accumulated-charge threshold gate for piezo release and test a seed `42` threshold-40 variant; lag-1 improves, but PSD and burst structure are not yet good enough to change defaults.
* Correct VLF/simulation burst comparison to use `burst_run_rate` instead of raw count for traces with different sample counts.
* Promote the best current seed `42` piezo candidate as the default: thresholded release, local receiver footprint, and sensor `5` for rendered VLF-like summaries.
* Validate tuned piezo defaults over seeds `40`, `41`, and `42`; best sensor varies by seed but scores are consistent.
* Separate piezo receiver locality controls from direct avalanche-signal receiver range, then regenerate seed `40`-`42` simulation CSVs.
* Refresh direct avalanche event lists, event maps, aligned synthetic model rows, tensors, and smoke reports after separating receiver ranges.
