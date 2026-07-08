# Next Actions

## Immediate Order

1. Keep self-supervised real VLF pretraining as the default modeling path while supervised labels are blocked.
2. Treat `trial-weekly-event-forecast.sh` as the current end-to-end event-list contract smoke test, not as a validated predictor.
3. Replace the trial forecast count/location heuristics with a swappable learned scorer trained first on synthetic aligned rows and calibrated against historical INGV rates.
4. Add a trial forecast map overlay showing current generated event coordinates over Italy for inspection.
5. Keep refreshing prospective INGV labels and real VLF-aligned rows until both classes exist, then compare learned multimodal forecasts against the trial baseline.

## Simulation

1. Keep accumulating Cumiana VLF captures before real multimodal model training claims.
2. Reduce sparse synthetic event-time clustering before promoting the refined sparse avalanche profile; current temporal sparse holdouts have zero positive test rows.
3. Reduce synthetic regime drift before expanding sequence model runs; regime-balanced evaluation helps debugging but is not a forecasting-style validation.
4. Add a small grouped-sensor piezo scan only if single-receiver traces prove too local after multi-seed validation.
5. Compare the tiny patch Transformer against the balanced GRU over additional synthetic seeds before increasing model size.
6. Optimize or chunk sequence materialization further before attempting substantially larger runs.
7. Compare the selected deeper patch Transformer against corrected-label GRU and tabular baselines after any material data change.
8. Use corrected-label missing-modality reports to decide whether VLF/piezo, direct avalanche, or combined sequence inputs deserve the next model pass.
9. Wait for real prospective rows to include both positive and negative labels before attempting real PyTorch training.
10. Use central-Italy historical seismic-only windows as the current real baseline smoke path.

## General

1. Clean and rationalize `docs/`: group overlapping notes, retire stale smoke-run documents, and keep a small index of current source, simulation, modeling, and operations docs.
2. Split `tests/test_acquisition_scaffold.py` by subsystem if test maintenance starts slowing down changes.
3. Reinstall/reload the updated prospective systemd unit if timer-managed image features and summaries are desired.
4. Keep the VLF capture and prospective timers running until the first target windows mature.
5. Continue periodic INGV refresh and prospective labeling as more target windows mature; train only after both classes exist in one table.
6. Add chunked sandpile snapshot storage only if larger pretraining runs outgrow `.npy` sanity snapshots.
7. Add slope/erosion smoothing to mountain-mode synthetic terrain if ridgeline-like visuals are needed.
8. Extend historical INGV backfill earlier than 2024 only if baseline windows need more class balance or longer seasonal coverage.
9. Validate Abelian Cumiana with a reproducible nonempty live Ogg, archive WAV, or archive VT pull; recent archive probes still returned zero usable bytes.
10. Add full rupture-mask outputs if synthetic event maps need spatial extent rather than centroid locations.
11. Generate a longer synthetic aligned dataset to reduce time-split distribution drift.
12. Use the piezo/Cumiana comparison report to tune only the piezo VLF mapping derived from `*.piezo.csv`.
13. Repeat mixed-domain VLF alignment after future Cumiana captures; require local-inlier selection to beat centroid and random controls before relying on it.
14. Shape direct avalanche signal events for INGV-like seismic event experiments without using the piezo/VLF path.
15. Use the signal-shape comparison report to tune simulation parameters separately for VLF-like and seismic-like outputs.
16. Add a small Markdown or CSV view over the compact model-run summary only if JSON becomes awkward to inspect.
17. Decode Abelian live/archive audio into time-frequency features after confirming sampling metadata and file readability.
18. Probe a wider Abelian archive range or alternate station only if a documented usable interval can be identified from the source pages.
19. Keep refreshing prospective INGV labels as VLF target windows mature; use real labels only after both target classes are present.
20. Run the selected deeper patch Transformer on the next larger synthetic diversity set after event sparsity and class balance look plausible.

## Completed

* Make self-supervised real VLF pretraining the default modeling path while supervised labels are one-class.
* Add `pretrain-sequence-autoencoder` and `pretrain-real-vlf-self-supervised.sh` for CPU masked sequence autoencoder pretraining.
* Run the first real Cumiana VLF self-supervised smoke: 247 rows, 224 windows, test masked MSE `0.835488` versus zero baseline `1.074356`.
* Add and run `score-real-vlf-anomaly-forecast.sh`; current label-free 7-day smoke forecast has demo probability `0.952514` for `2026-07-06T06:50:50Z` to `2026-07-13T06:50:50Z`.
* Add and run `trial-weekly-event-forecast.sh`; the current `2026-07-08` trial emits 25 capped `>M2` event-coordinate rows for `2026-07-08` to `2026-07-15`, using INGV history, real VLF context, astronomy captures, and synthetic avalanche artifacts.
* Switch `compare-vlf-embedding-domains.sh` to shape-profile descriptors; held-out real masked reconstruction now beats baseline by about `6.8%`, and synthetic masked reconstruction also edges baseline.
* Add synthetic VLF inlier marking to the embedding-domain diagnostic; the current 25% inlier subset keeps 14,983 windows and sharply reduces embedding scale mismatch.
* Add and run `evaluate-vlf-synthetic-inlier-transfer.sh`; synthetic-inlier pretraining reconstructs held-out real VLF descriptors better than the zero baseline, but transfer embedding centroids remain far apart.
* Add and run `evaluate-vlf-mixed-domain-alignment.sh`; CORAL-aligned mixed real/synthetic training improves held-out embedding centroid distance to `1.033580`, with centroid and random controls close enough to require further validation.
* Add `transform-piezo-signal` and run `sweep-piezo-vlf-alignment.sh`; burst/high-pass variants improve short-run embedding distance but currently worsen held-out real reconstruction.
* Refresh prospective labels to 55 matured rows per scope and rebuild real VLF-aligned model inputs; all-Italy remains 55 positive / 0 negative and central Italy remains 0 positive / 55 negative.
* Add `estimate-model-scale` and `estimate-model-scale.sh` to capture larger-model gates, sequence sizes, and CPU-only model guidance.
* Add `max_events` to avalanche event-extraction tuning, add a stable sparse-event `shape_score`, and identify refined central-Italy sparse profiles.
* Correct aligned synthetic target semantics so `target_horizon_rows` means the next `N` complete future rows, then rebuild default 20000-step model artifacts.
* Add and run `sweep-sparse-target-horizon.sh`; sparse positive labels now scale from `3/501` at horizon `1` to `72/432` at horizon `24`, but temporal test splits still have no positives.
* Rerun tabular PyTorch, sequence GRU, temporal diagnostics, and tabular-vs-sequence comparison under corrected-label targets.
* Add and run `train-deep-patch-transformer.sh`; selected deeper patch Transformer pretrains on synthetic sequence data and records real fine-tune readiness, currently blocked by one-class real labels.
* Extend INGV historical backfill from `2024-01-01` through `2026-07-07`, producing 4836 all-Italy events, 594 central-Italy events, and 130 weekly labeled windows per scope.
* Rebuild seismic-only temporal baselines and real-vs-synthetic signal-shape diagnostics from the extended INGV history.
* Add `docs/model-scaling-requirements.md`; current real VLF rows are blocked, while the full synthetic 20000-step table can support only a tiny synthetic-only larger-model check.
* Add and run a pre-relabel tiny CPU patch Transformer split-holdout path; best regime-balanced calibrated row was `0.637500` for `sequence_piezo_vlf_only`, below the balanced GRU `sequence_full` score.
* Add deterministic regime-balanced split assignment and explicit split sequence evaluation.
* Add and run `train-sequence-full-balanced.sh`; post-burn-in `sequence_full` reached calibrated balanced accuracy `0.650000` on the balanced synthetic split.
* Refresh prospective labels to 23 matured rows per scope and rebuild real VLF-aligned model inputs; real training remains class-blocked.
* Fix `refresh-prospective-labels.sh` so prospective combined INGV files exclude earlier historical backfill chunks.
* Add and run `compare-real-synthetic-models.sh`, producing compact JSON/CSV real-vs-synthetic model comparisons.
* Add `docs/model-comparison.md` with the current central-Italy seismic baseline, synthetic sequence, and post-burn-in regime interpretation.
* Add and run `backfill-ingv-history.sh`; 2026 historical INGV backfill produced ready seismic-only windows for all-Italy and central Italy.
* Add burn-in/regime annotation for synthetic aligned rows and run a post-burn-in `sequence_full` regime robustness check.
* Add sequence evaluation filtering so focused robustness scripts can run only `sequence_full`.
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
* Rerun corrected-label CPU PyTorch tabular and sequence reports; temporal rows remain weak, while seed holdouts range from `0.720690` to `0.821105`.
* Add and run a pre-relabel CPU PyTorch tabular MLP on the `20000`-step multi-seed synthetic aligned table; best calibrated chronological balanced accuracy was `0.543076`.
* Add and run pre-relabel CPU PyTorch leave-one-seed-out tabular MLP reports; best calibrated balanced accuracy ranged from `0.752810` to `0.768286` across held-out seeds.
* Add a demo map overlay for actual avalanche-derived events and PyTorch predicted-positive target-window event hits.
* Add a shared model feature-role registry so real VLF fields and synthetic piezo/VLF analogue fields can feed the PyTorch VLF path.
* Add a first CPU PyTorch GRU sequence model over materialized synthetic avalanche, piezo/VLF, and summary tensors.
* Run pre-relabel temporal and leave-one-seed-out sequence GRU reports; chronological balanced accuracy was flat at `0.500000`, while seed holdouts reached `0.712754` to `0.772558`.
* Add a comparison command and wrapper script for tabular-vs-sequence model-run summaries.
* Add a bounded sequence GRU sweep script for lookback and hidden-unit checks.
* Add a missing-modality sequence smoke script for VLF/piezo-only and no-piezo sequence checks.
* Add a real Cumiana VLF image sequence materialization script matching the synthetic sequence manifest shape.
* Add a prospective INGV refresh-and-label script for matured VLF image windows.
* Run the corrected-label tabular-vs-sequence comparison; best calibrated row is `0.826389` for `seismic_vlf_unified` on held-out `seed41`.
* Run the pre-relabel tabular-vs-sequence comparison; best calibrated row was `0.772558` for sequence piezo/VLF-only on held-out `seed42`.
* Run a tiny sequence sweep smoke; best calibrated row is `0.709624` for `sequence_full` on held-out `seed41`.
* Run a missing-modality seed42 smoke; no-piezo direct avalanche outperformed piezo-only in this short check.
* Materialize current real Cumiana VLF image features into a `247 x 1 x 25` sequence manifest.
* Refresh INGV labels through `2026-07-08`; real prospective rows still lack class variation.
* Split Abelian VLF acquisition into common, live-stream, archive/probe, and compatibility modules; no production Python file is now over 500 lines.
* Run the full sequence sweep over lookbacks `30`, `60`, and `120`, hidden sizes `16` and `24`, and `10` epochs.
* Add nested comparison support so sweep comparison rows can be included in family-level comparisons.
* Generate pre-relabel `data/derived/models/model_family_comparison.json`; best calibrated row was `0.772558` for sequence piezo/VLF-only on held-out `seed42`.
* Add and run `prepare-real-model-inputs.sh`; real aligned VLF tables are scaffolded, but both still have insufficient class variation.
* Add and run sequence modality diagnostics over `112` evaluation rows; the direct-only sweep result is not a sufficient reason to change defaults because epoch counts differ and temporal rows remain near `0.5`.
* Add and run pre-relabel `matched-sequence-comparison.sh`; the matched 20-epoch comparison kept `sequence_piezo_vlf_only`, `lookback=60`, `hidden=24` as the strongest single row at `0.772558`, while `sequence_full` had the best mean group score.
* Add `summarize-sequence-selection` and run the matched sequence selection report.
* Add and run `repeat-sequence-training-seeds.sh`; piezo/VLF-only remains the best single row, while `sequence_full` wins mean group and worst-held-out-seed stability.
* Add `train-real-tabular-model.sh`; it correctly refuses to train while all-Italy has `18` positives and `0` negatives.
* Refresh INGV labels and real aligned model inputs again; real training remains blocked by insufficient class variation.
* Add `train-real-deep-patch-transformer.sh`; current all-Italy and central-Italy fine-tune reports correctly block on one-class labels.
* Refresh prospective labels to 54 matured rows per scope and rebuild real VLF-aligned model inputs.
* Rerun missing-modality and sequence diagnostics; direct avalanche remains strongest on grouped synthetic checks, while piezo/VLF-only still carries signal.
* Add and smoke-test `run-synthetic-diversity-smoke.sh` on two 128x128, 1000-step seeds with evaluations disabled.
