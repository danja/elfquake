# Next Actions

## Simulation

1. Backfill more INGV event windows and keep accumulating Cumiana VLF captures before model training claims.
2. Compare longer synthetic runs against real seismic/VLF shape metrics before treating the generator as stable.
3. Add burn-in or regime-balanced synthetic split handling before treating chronological synthetic model metrics as stable.
4. Add a small grouped-sensor piezo scan only if single-receiver traces prove too local after multi-seed validation.
5. Optimize or chunk sequence materialization further before attempting substantially larger runs.
6. Review the tabular-vs-sequence comparison and bounded sequence sweep outputs before changing the default GRU lookback.
7. Use missing-modality reports to decide whether VLF/piezo, direct avalanche, or combined sequence inputs deserve the next model pass.
8. Wait for real prospective rows to include both positive and negative labels before attempting real PyTorch training.
9. Use `sequence_full` for the next robustness experiments, while keeping piezo/VLF-only noted as the strongest single synthetic group row.
10. Add a regime-balanced or burn-in-aware synthetic split before spending more time on chronological sequence metrics.

## General

1. Split `tests/test_acquisition_scaffold.py` by subsystem if test maintenance starts slowing down changes.
2. Reinstall/reload the updated prospective systemd unit if timer-managed image features and summaries are desired.
3. Keep the VLF capture and prospective timers running until the first target windows mature.
4. Continue periodic INGV refresh and prospective labeling as more target windows mature; train only after both classes exist in one table.
5. Add chunked sandpile snapshot storage only if larger pretraining runs outgrow `.npy` sanity snapshots.
6. Add slope/erosion smoothing to mountain-mode synthetic terrain if ridgeline-like visuals are needed.
7. Backfill enough historical INGV windows to get both positive and negative target classes by region.
8. Validate Abelian Cumiana with a reproducible nonempty live Ogg, archive WAV, or archive VT pull; recent archive probes still returned zero usable bytes.
9. Add full rupture-mask outputs if synthetic event maps need spatial extent rather than centroid locations.
10. Generate a longer synthetic aligned dataset to reduce time-split distribution drift.
11. Use the piezo/Cumiana comparison report to tune only the piezo VLF mapping derived from `*.piezo.csv`.
12. Shape direct avalanche signal events for INGV-like seismic event experiments without using the piezo/VLF path.
13. Use the signal-shape comparison report to tune simulation parameters separately for VLF-like and seismic-like outputs.
14. Add a small Markdown or CSV view over the compact model-run summary only if JSON becomes awkward to inspect.
15. Decode Abelian live/archive audio into time-frequency features after confirming sampling metadata and file readability.
16. Probe a wider Abelian archive range or alternate station only if a documented usable interval can be identified from the source pages.
17. Keep refreshing prospective INGV labels as VLF target windows mature; use real labels only after both target classes are present.
18. Add a real PyTorch training wrapper only after real aligned tables have both target classes.

## Completed

* Add repo-local Codex skills for source ingest, prospective data refresh, simulation pipeline runs, and synthetic model workflows.
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
* Add `refresh-synthetic-model-artifacts.sh` to reproduce synthetic event, map, aligned row, tensor, and smoke-report refreshes from existing simulation CSVs.
* Regenerate current-default `20000`-step seeds `40`, `41`, and `42`, then refresh longer-run event lists, maps, aligned tensors, and smoke reports.
* Optimize aligned window aggregation with timestamp indexes so longer sequence refreshes do not repeatedly scan every record for every window.
* Split signal-shape metrics, piezo signal/audio helpers, and sandpile output helpers out of previously oversized production modules.
* Refactor the CLI into command-family modules while preserving the existing command names and error handling.
* Add optional dependency constraints so Numba-compatible installs keep NumPy below `2.5`.
* Add temporal split diagnostics and use them to explain the weak `20000`-step chronological synthetic holdout.
* Add Abelian Cumiana live Ogg capture, archive retrieve scaffolding, and audio/container feature extraction.
* Add a reproducible Abelian archive probe CSV report; first tested Cumiana `wav`/`vt` rows were HTTP 200 but zero usable bytes.
* Probe Abelian Cumiana archive availability across five additional timestamps from `2026-06-29` through `2026-07-05`; all ten `wav`/`vt` rows were HTTP 200 but zero usable bytes.
* Refresh INGV through the July rollover, rebuild 247-row prospective VLF image tables for central Italy and all Italy, and confirm no missing VLF, VLF image, or astronomy coverage.
* Label the first matured prospective VLF target window: central Italy was negative, all Italy was positive, with one labeled row per table.
* Add and run a CPU PyTorch tabular MLP on the `20000`-step multi-seed synthetic aligned table; best calibrated chronological balanced accuracy is now `0.543076`.
* Add and run CPU PyTorch leave-one-seed-out tabular MLP reports; best calibrated balanced accuracy now ranges from `0.752810` to `0.768286` across held-out seeds.
* Add a demo map overlay for actual avalanche-derived events and PyTorch predicted-positive target-window event hits.
* Add a shared model feature-role registry so real VLF fields and synthetic piezo/VLF analogue fields can feed the PyTorch VLF path.
* Add a first CPU PyTorch GRU sequence model over materialized synthetic avalanche, piezo/VLF, and summary tensors.
* Run temporal and leave-one-seed-out sequence GRU reports; chronological balanced accuracy is flat at `0.500000`, while seed holdouts reach `0.712754` to `0.772558`.
* Add a comparison command and wrapper script for tabular-vs-sequence model-run summaries.
* Add a bounded sequence GRU sweep script for lookback and hidden-unit checks.
* Add a missing-modality sequence smoke script for VLF/piezo-only and no-piezo sequence checks.
* Add a real Cumiana VLF image sequence materialization script matching the synthetic sequence manifest shape.
* Add a prospective INGV refresh-and-label script for matured VLF image windows.
* Run the current tabular-vs-sequence comparison; best calibrated row is `0.772558` for sequence piezo/VLF-only on held-out `seed42`.
* Run a tiny sequence sweep smoke; best calibrated row is `0.709624` for `sequence_full` on held-out `seed41`.
* Run a missing-modality seed42 smoke; no-piezo direct avalanche outperformed piezo-only in this short check.
* Materialize current real Cumiana VLF image features into a `247 x 1 x 25` sequence manifest.
* Refresh INGV labels through `2026-07-08`; real prospective rows still lack class variation.
* Split Abelian VLF acquisition into common, live-stream, archive/probe, and compatibility modules; no production Python file is now over 500 lines.
* Run the full sequence sweep over lookbacks `30`, `60`, and `120`, hidden sizes `16` and `24`, and `10` epochs.
* Add nested comparison support so sweep comparison rows can be included in family-level comparisons.
* Generate `data/derived/models/model_family_comparison.json`; best calibrated row remains `0.772558` for sequence piezo/VLF-only on held-out `seed42`.
* Add and run `prepare-real-model-inputs.sh`; real aligned VLF tables are scaffolded, but both still have insufficient class variation.
* Add and run sequence modality diagnostics over `112` evaluation rows; the direct-only sweep result is not a sufficient reason to change defaults because epoch counts differ and temporal rows remain near `0.5`.
* Add and run `matched-sequence-comparison.sh`; the matched 20-epoch comparison keeps `sequence_piezo_vlf_only`, `lookback=60`, `hidden=24` as the strongest single row at `0.772558`, while `sequence_full` has the best mean group score.
* Add `summarize-sequence-selection` and run the matched sequence selection report.
* Add and run `repeat-sequence-training-seeds.sh`; piezo/VLF-only remains the best single row, while `sequence_full` wins mean group and worst-held-out-seed stability.
* Add `train-real-tabular-model.sh`; it correctly refuses to train while all-Italy has `18` positives and `0` negatives.
* Refresh INGV labels and real aligned model inputs again; real training remains blocked by insufficient class variation.
