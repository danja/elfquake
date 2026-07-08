# Next Actions

## Immediate Priority

1. Treat `trial-weekly-event-forecast.sh` as the current end-to-end event-list contract smoke test, not as a validated predictor.
2. Improve the synthetic-trained learned scorer: current held-out balanced accuracy is only `0.507576`, with poor negative recall.
3. Replace the shared event placement heuristics with model-informed spatial/event-count adapters while preserving the forecast CSV contract.
4. Keep self-supervised real VLF pretraining as the default real-data modeling path while supervised VLF-aligned labels remain one-class or sparse.
5. Continue periodic INGV refresh and prospective relabeling; latest real aligned rows are still one-class (`69/0` all-Italy, `0/69` central Italy).

## Modeling

1. Calibrate weekly event counts against historical INGV `>M2` rates before trusting any neural score scale.
2. Add a model-informed spatial adapter so learned scores can alter event locations without breaking the event-list contract.
3. Add a comparison report between heuristic trial forecasts and learned-scorer forecasts.
4. Keep direct avalanche-derived seismic features separate from piezo/VLF-like features; use ablations to test their contribution independently.
5. Rerun the selected deeper patch Transformer only after synthetic event sparsity and target balance are improved enough for meaningful temporal checks.

## Data

1. Keep accumulating Cumiana VLF image captures and refreshing image features.
2. Refresh prospective INGV labels as target windows mature; train supervised real models only after one table has both positive and negative labels.
3. Validate Abelian Cumiana live/archive audio only if a reproducible nonempty pull is found; current probes returned zero usable bytes.
4. Extend historical INGV backfill earlier than 2024 only if weekly baseline calibration needs longer seasonal coverage.
5. Repeat mixed real/synthetic VLF alignment after new Cumiana captures; require improvements over centroid and random controls before relying on inlier selection.

## Simulation

1. Reduce sparse synthetic event-time clustering before promoting the refined sparse avalanche profile.
2. Generate a longer and more diverse synthetic aligned dataset once event sparsity and class balance look plausible.
3. Tune direct avalanche event extraction for INGV-like seismic event shapes without using the piezo/VLF path.
4. Tune the piezo/VLF mapping only from `*.piezo.csv` and compare against Cumiana VLF shape reports.
5. Add full rupture-mask outputs only if map demos need spatial extent rather than centroid locations.

## Maintenance

1. Keep docs concise: one current source doc, one simulation doc, one modeling doc, one operations/steps doc, and one report.
2. Split `tests/test_acquisition_scaffold.py` by subsystem if test maintenance starts slowing changes.
3. Add chunked sandpile snapshot storage only if larger pretraining runs outgrow current `.npy` sanity snapshots.
4. Keep optional dependencies CPU-compatible on this system; do not add GPU-only paths.

## Recent Completed

* Added and ran `trial-weekly-event-forecast.sh`; the current `2026-07-08` trial emits 25 capped `>M2` event-coordinate rows for `2026-07-08` to `2026-07-15`.
* Added and ran `learned-weekly-event-forecast.sh`; it trains a synthetic-window logistic scorer and emits the same weekly event-list CSV contract.
* Added learned-scorer metadata to the forecast report without changing the CSV event-row contract.
* Refreshed INGV prospective labels and rebuilt real model inputs; both scopes now have 69 labeled rows but remain class-blocked.
* Added and ran `trial-forecast-map.sh`, rendering `data/derived/maps/mag_gt2_weekly_trial_forecast_map.png` from the trial forecast CSV.
* Added `docs/forecast-interface.md` to define the stable weekly event-list output contract for trial and future learned scorers.
* Added `docs/output-example.md` with the top three highest-magnitude trial rows and nearest mapped places.
* Added self-supervised real VLF pretraining and label-free anomaly scoring as the default real-data development path while labels are sparse.
* Built current real VLF-aligned all-Italy and central-Italy model inputs; both remain class-blocked for supervised real training.
* Extended INGV historical backfill from `2024-01-01` through `2026-07-07`, producing historical seismic baseline windows.
* Added synthetic aligned sequence/tensor paths, GRU and patch-Transformer smoke models, missing-modality checks, and model-run summaries.
* Added direct avalanche-derived event extraction, synthetic event maps, and separate piezo/VLF-like signal outputs.
* Added repo-local Codex skills for source ingest, data refresh, simulation, and synthetic modeling workflows.
