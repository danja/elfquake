# Next Actions

## Immediate Priority

1. Treat `./scripts/trial-weekly-event-forecast.sh` as the current end-to-end event-list contract smoke test, not as a validated predictor.
2. Use `./scripts/train-synthetic-event-list-patch-transformer.sh` as the current Transformer synthetic-pretraining smoke: the best short run is lookback `12`, patch `3`, dropout `0.1`, with piezo/VLF-only calibrated balanced accuracy `0.608629`.
3. Stabilize event-list neural heads before forecast promotion: sequence heads and patch Transformers both show useful piezo/VLF signal, but neither has passed repeated-seed and longer-episode checks.
4. Keep self-supervised real VLF pretraining as the default real-data modeling path while supervised VLF-aligned labels remain one-class or sparse.
5. Continue periodic INGV refresh and prospective relabeling; latest real aligned rows are still one-class (`69/0` all-Italy, `0/69` central Italy).

## Modeling

1. Use `./scripts/sweep-synthetic-event-list-patch-transformer.sh` after material simulation or target changes; keep piezo/VLF-only, direct-avalanche-only, combined, and full ablations in every run.
2. Calibrate weekly event counts against historical INGV `>M2` rates before trusting any neural score scale.
3. Compare every weekly forecast run with `./scripts/compare-weekly-forecasts.sh` and track Stage 1/Stage 2 pass/fail status.
4. Keep direct avalanche-derived seismic features separate from piezo/VLF-like features; use ablations to test their contribution independently.
5. Do not make validation-selected thresholding or early stopping the default; both underperformed. Next, reduce variance through larger episode batches, then rerun the sequence-head and patch-Transformer sweeps.

## Data

1. Keep accumulating Cumiana VLF image captures and refreshing image features.
2. Refresh prospective INGV labels as target windows mature; train supervised real models only after one table has both positive and negative labels.
3. Validate Abelian Cumiana live/archive audio only if a reproducible nonempty pull is found; current probes returned zero usable bytes.
4. Extend historical INGV backfill earlier than 2024 only if weekly baseline calibration needs longer seasonal coverage.
5. Repeat mixed real/synthetic VLF alignment after new Cumiana captures; require improvements over centroid and random controls before relying on inlier selection.

## Simulation

1. Run `./scripts/run-longer-synthetic-transformer-batch.sh` when CPU time is available, then validate with `./scripts/validate-synthetic-event-list-drift.sh` and rerun the Transformer sweep.
2. Improve event-list weighting, sequence context, or temporal validation strategy; denser direct-event extraction alone made labels healthier but did not improve model smoke metrics.
3. Compare future episode-batch h6 drift against the current scaled `WARMUP_STEPS=3000` delta `0.187025`.
4. Revisit structured initial fill only with delayed bottom-layer removal; the first fill probe drifted at `0.307937`.
5. Tune the piezo/VLF mapping only from `*.piezo.csv` and compare against Cumiana VLF shape reports.

## Maintenance

1. Keep docs concise: one current source doc, one simulation doc, one modeling doc, one operations/steps doc, and one report.
2. Split `tests/test_acquisition_scaffold.py` by subsystem if test maintenance starts slowing changes.
3. Add chunked sandpile snapshot storage only if larger pretraining runs outgrow current `.npy` sanity snapshots.
4. Keep optional dependencies CPU-compatible on this system; do not add GPU-only paths.

## Recent Completed

* Added synthetic event-list target generation from avalanche event CSVs, including future count, occurrence, magnitude, centroid, and time-to-first-event fields.
* Added dependency-light synthetic event-list heads for occurrence, count, max magnitude, and centroid. The h6 balanced split reaches balanced accuracy `0.887566`, count MAE `0.506783`, and centroid median error `145.585806 km`; the temporal split still fails with balanced accuracy `0.500000`.
* Added drift diagnostics, episode annotation, and a validation wrapper. Current h6 temporal split has train positive rate `0.318653` and test positive rate `0.927835`; episode-balanced validation reaches balanced accuracy `0.878079`.
* Added `./scripts/run-synthetic-episode-batch.sh` for shorter stationarity-tuned synthetic episodes with localized sources preserved.
* Ran two stationarity profiles. The first six-episode 5000-step batch still drifted badly (`0.652766` positive-rate delta). The aggressive three-episode 3000-step probe fixed drift (`0.065608` delta, warning `ok`) but is too small for model selection.
* Added structured initial fill and ran a three-seed probe. It started loaded and stayed near mean height `3.2`, but h6 drift was `0.307937`, worse than the no-prefill aggressive profile.
* Added unrecorded sandpile warm-up. The first three-seed `WARMUP_STEPS=1000` probe improved h6 drift to `0.017989` with warning `ok`.
* Scaled `WARMUP_STEPS=1000` to nine episodes; drift failed again (`0.294146` delta), showing the small result did not scale.
* Tested `WARMUP_STEPS=3000` on three seeds. It passed h6 drift (`0.048677`, warning `ok`) and the temporal event-list smoke model reached balanced accuracy `0.674342`; this is the new episode-batch default but still needs scaling.
* Scaled `WARMUP_STEPS=3000` to nine episodes. The run stayed drift-ok (`0.187025` delta) with 396 labeled rows, but temporal model balanced accuracy was only `0.468045`.
* Tested denser direct avalanche extraction on the same nine episodes. `_q099_w60_m10` saturated h6 labels (`0.828283` positive rate), while `_q0995_w120_m5` improved class balance (`0.512626`) but did not beat the sparse default model checks.
* Added deterministic feature-bag ensembles to the dependency-light event-list occurrence head. The default script now uses 8 members with 50% feature bags, improving the scaled sparse temporal/balanced/episode-balanced checks to `0.498120`, `0.616473`, and `0.607551`.
* Added richer synthetic event-list targets for event-rate, log magnitude energy, early/middle/final horizon counts, peak timing, event duration, and spatial spread.
* Added regression shape heads for the richer event-list targets and report metrics for each head. On the balanced split, event-rate MAE is `0.068936`, early/middle/final count MAE is `0.180167`/`0.246829`/`0.194170`, and spatial-spread MAE is `20.317859 km`.
* Added an optional boosted-stump occurrence head. It improves the balanced engineering split to `0.629173` balanced accuracy, but fails the chronological split at `0.347744`; keep the feature-bag logistic ensemble as the default.
* Added `./scripts/probe-synthetic-event-list-models.sh` and `summarize-synthetic-event-list-probes` to make horizon, burn-in, feature-cap, ensemble-size, boosted-stump, and balanced-control checks reproducible.
* Smoke-tested the probe harness with `HORIZONS=3`, `BURN_IN_FRACTIONS=0`, `RUN_BALANCED_CONTROLS=0`, and `EPOCHS=20`; it wrote comparable summaries and confirmed the harness works end to end.
* Ran the full synthetic event-list probe. The best drift-ok chronological result was h6 with a 32-feature cap at balanced accuracy `0.511278`; the best drift-ok balanced control was h12 boosted stumps at `0.651543`.
* Updated probe summaries so model rows include matching target drift warning and positive-rate delta, making drift-ok temporal results easy to separate from suspect runs.
* Added `build-synthetic-lagged-context` and `./scripts/probe-synthetic-event-list-lagged-context.sh` to test explicit recent-history features without target leakage.
* Ran lagged-context h6 probes. The best drift-ok chronological result was the 256-feature cap at balanced accuracy `0.587093`, just below the synthetic gate; all features regressed to `0.530702`.
* Added `train-synthetic-event-list-sequence-head` and `./scripts/train-synthetic-event-list-sequence-head.sh`. The h6 lookback-12 seed-42 sequence head reaches calibrated balanced accuracy `0.609649`, but seed/config checks range from `0.467419` to `0.603383`, so it needs stability work before promotion.
* Added `./scripts/sweep-synthetic-event-list-sequence-head.sh` and `summarize-synthetic-event-list-sequence-heads`. The first 18-run sweep selected lookback `12`, dropout `0.1` as the best mean config: mean `0.600459`, min `0.576441`, max `0.645363`.
* Added `ensemble-synthetic-event-list-sequence-heads` and `./scripts/ensemble-synthetic-event-list-sequence-head.sh`. The three-seed ensemble scored `0.591479`; pairwise `42+99` scored `0.644110`, showing useful but selection-sensitive variance reduction.
* Added validation-selected and early-stopped sequence-head controls. Validation-selected lookback-12/dropout-0.1 averaged `0.539265`; early-stopped dropout-0.1 averaged `0.494570`, so neither should be promoted.
* Added `prepare-transformer-target-input`, `./scripts/train-synthetic-event-list-patch-transformer.sh`, and `./scripts/sweep-synthetic-event-list-patch-transformer.sh` to train the patch Transformer directly on the richer h6 event-list target table.
* Ran a four-config h6 patch-Transformer sweep on the warmed nine-episode synthetic data. Piezo/VLF-only was the strongest ablation in every config; best short-run calibrated balanced accuracy was `0.608629` with lookback `12`, patch `3`, dropout `0.1`.
* Added `./scripts/run-longer-synthetic-transformer-batch.sh` as the repeatable CPU-only route for larger warmed synthetic batches before Transformer retuning.
* Added and ran `./scripts/trial-weekly-event-forecast.sh`; the current `2026-07-08` trial emits 25 capped `>M2` event-coordinate rows for `2026-07-08` to `2026-07-15`.
* Added and ran `./scripts/learned-weekly-event-forecast.sh`; it trains a synthetic-window logistic scorer and emits the same weekly event-list CSV contract.
* Added learned-scorer metadata to the forecast report without changing the CSV event-row contract.
* Added `docs/success-criteria.md` with staged scaffold, synthetic utility, real readiness, and prediction-claim gates.
* Added and ran `./scripts/compare-weekly-forecasts.sh`; Stage 1 event-contract criteria pass, Stage 2 synthetic-model criteria fail.
* Refreshed INGV prospective labels and rebuilt real model inputs; both scopes now have 69 labeled rows but remain class-blocked.
* Added and ran `./scripts/trial-forecast-map.sh`, rendering `data/derived/maps/mag_gt2_weekly_trial_forecast_map.png` from the trial forecast CSV.
* Added `docs/forecast-interface.md` to define the stable weekly event-list output contract for trial and future learned scorers.
* Added `docs/output-example.md` with the top three highest-magnitude trial rows and nearest mapped places.
* Added self-supervised real VLF pretraining and label-free anomaly scoring as the default real-data development path while labels are sparse.
* Built current real VLF-aligned all-Italy and central-Italy model inputs; both remain class-blocked for supervised real training.
* Extended INGV historical backfill from `2024-01-01` through `2026-07-07`, producing historical seismic baseline windows.
* Added synthetic aligned sequence/tensor paths, GRU and patch-Transformer smoke models, missing-modality checks, and model-run summaries.
* Added direct avalanche-derived event extraction, synthetic event maps, and separate piezo/VLF-like signal outputs.
* Added repo-local Codex skills for source ingest, data refresh, simulation, and synthetic modeling workflows.
