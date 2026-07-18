# Next Actions

## Immediate Priority

1. Run `./scripts/run-real-transfer-trial.sh` after each INGV refresh. It is the first chronological 80/20 real-data check at M2.5, expressed as seven-day fixed Italy spatial-cell targets; interpret its held-out map and scores only as an experimental baseline.
2. Treat `./scripts/trial-weekly-event-forecast.sh` as the current end-to-end event-list contract smoke test, not as a validated predictor.
2. Use `./scripts/evaluate-piezo-group-holdout.sh` as the primary synthetic stability check. Its fixed three-seed ensemble averages `0.632634`, but passes both recall floors on only 6 of 9 episodes.
3. Keep random-init piezo/VLF-only as the leading controlled Transformer architecture. It averages `0.619033` within episodes, but has not passed unseen-episode stability; direct and summary branches remain disabled by default.
4. Keep label-free real VLF pretraining as the default real-data path while supervised VLF-aligned labels remain one-class or sparse; require reconstruction to beat both zero and last-patch baselines.
5. Continue periodic INGV refresh and prospective relabeling; the current catalog-coverage guard prevents false maturity. Region-level tables remain one-class, so use `./scripts/prepare-italy-spatial-model-inputs.sh` for the fixed-cell baseline until more temporal coverage arrives.
6. Repeat `./scripts/evaluate-italy-spatial-baseline.sh` after each refresh; its `--group-by-time` split keeps all cells from one VLF window in the same partition.
7. Do not promote the spatial model from the current cell holdout: only 5 of 19 cells contain positive labels, so leave-one-cell-out evaluation is mostly one-class. Accumulate more time coverage before using this as a transfer test.
8. Treat the permutation result as a stop signal for interpretation: the five timestamp-shuffled controls averaged `0.679362`, above the real-order score `0.655320`. Do not tune the model against this table until the live capture history is substantially longer.

## Modeling

1. Run `./scripts/run-transfer-experiments.sh` after each real-data refresh. It compares historical rate, real-only random initialization, synthetic transfer, rolling-origin folds, and a train-only grid selection before one final holdout evaluation. The default synthetic corpus now includes four long episodes; add more 20,000-step episodes before treating transfer changes as stable.
1. Generate more independent warmed episodes and rerun leave-one-episode-out evaluation; nine episodes are not enough to estimate regime robustness tightly.
2. Do not add the default piezo potential channel to model training yet. Its spatial average failed a nine-episode causal lead-time check; event-nearest diagnostics are positive but use future event locations and are not valid inputs.
3. Calibrate weekly event counts against historical INGV `>M2` rates before trusting any neural score scale.
4. Compare every weekly forecast run with `./scripts/compare-weekly-forecasts.sh` and track Stage 1/Stage 2 pass/fail status.
5. Keep direct avalanche-derived seismic features separate from piezo/VLF-like features; use ablations to test their contribution independently.

## Data

1. Keep accumulating Cumiana VLF image captures and refreshing image features.
2. Refresh prospective INGV labels as target windows mature; train supervised real models only after one table has both positive and negative labels.
3. Validate Abelian Cumiana live/archive audio only if a reproducible nonempty pull is found; current probes returned zero usable bytes.
4. Extend historical INGV backfill earlier than 2024 only if weekly baseline calibration needs longer seasonal coverage.
5. Repeat mixed real/synthetic VLF alignment after new Cumiana captures; require improvements over centroid and random controls before relying on inlier selection.

## Simulation

1. Run `./scripts/run-longer-synthetic-transformer-batch.sh` when CPU time is available, validate drift, then rerun `./scripts/evaluate-piezo-group-holdout.sh` against the larger episode set.
2. Keep `damage_total` as a validated synthetic precursor diagnostic, not a default Transformer feature. A matched nine-fold screen regressed from `0.599648` without damage channels to `0.586848` with them.
3. Keep the duration-aligned `SOURCE_COUNT=64`, refill `470`, removal interval `20`, and `q=0.998/window=120` profile as a valid synthetic target baseline (`47.0%` positives, temporal drift `0.182`). It has no confirmed piezo lead and is not a precursor-training profile.
4. The first two-stage mature-weakness profile failed its nine-episode causal confirmation despite stable target drift. Do not tune its scalar parameters immediately or train a model. Document a stronger physical mechanism proposal, such as a spatially propagating rupture/nucleation state, before another synthetic dynamics run.
5. Compare future episode-batch h6 drift against the current scaled `WARMUP_STEPS=3000` delta `0.187025`.
6. Revisit structured initial fill only with delayed bottom-layer removal; the first fill probe drifted at `0.307937`.
7. Tune the piezo/VLF mapping only from `*.piezo.csv` and compare against Cumiana VLF shape reports.

## Maintenance

1. Keep docs concise: one current source doc, one simulation doc, one modeling doc, one operations/steps doc, and one report.
2. Split `tests/test_acquisition_scaffold.py` by subsystem if test maintenance starts slowing changes.
3. Add chunked sandpile snapshot storage only if larger pretraining runs outgrow current `.npy` sanity snapshots.
4. Keep optional dependencies CPU-compatible on this system; do not add GPU-only paths.

## Recent Completed

* Added `run-transfer-experiment-suite` and `./scripts/run-transfer-experiments.sh`. The suite performs matched real-only versus synthetic-pretrained ablations, four rolling-origin folds, and a train-only threshold/grid selection followed by final holdout evaluation.
* Generated a 20,000-step CPU episode at seed `4300` and stacked its dense synthetic records with seeds `40`--`42`. Synthetic episodes are offset in synthetic time during training because their source timestamps intentionally share one demonstration origin; real timestamps are unchanged.
* The expanded transfer corpus contains 79,976 synthetic records and 190 weekly spatial samples. Transfer remains below the historical spatial-rate baseline on precision and is not predictive evidence.
* Added modality-specific Transformer patch adapters, elapsed-time inputs, separate observed/corruption/padding masks, masked reconstruction, modality dropout, frozen probes, compatible checkpoint transfer, and missing-modality checks.
* Added stable name-derived Transformer initialization. Adding or reordering unused modality adapters no longer changes shared weights or advances the global PyTorch RNG.
* Reran seven transfer regimes under controlled initialization. Random-init piezo/VLF-only is strongest at mean `0.619033`; synthetic pretraining falls to `0.534272`, so current self-supervision has no demonstrated downstream gain.
* Reran late gated fusion under the same control. Random anchored full and direct-only variants reach `0.609241` and `0.606385`, but neither beats the piezo/VLF anchor and direct improves when disabled.
* Added and ran nine-episode by three-seed leave-one-episode-out evaluation. Mean balanced accuracy is `0.578712`, the worst fold is `0.275641`, and 14 of 27 folds meet both recall floors.
* Added a fixed three-seed probability ensemble with training-only threshold calibration. It raises mean unseen-episode balanced accuracy to `0.632634`, but only 6 of 9 episodes meet both recall floors; a fixed threshold passes the same six.
* Diagnosed episode instability: the same piezo features change class-effect sign between trajectories, while h6 labels commonly form six-row positive runs against only 12 minutes of model context.
* Tested h3 labels, h6 with 60-minute context, and opt-in spatial sensor aggregates. Their ensemble means are `0.580991`, `0.512103`, and `0.544096`; all trail the h6/12-minute mean-only baseline.
* Added `compare-piezo-group-holdouts` and `./scripts/summarize-piezo-group-holdouts.sh`; zero of four current variants passes both the mean and recall-stability gates.
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
* Fixed prospective-label maturity so a target is only labeled when the event catalog covers its complete horizon. The incremental refresh now rebuilds existing candidate rows, uses stable current catalog paths, and caches unchanged VLF image features.
* Added causal pre-relaxation piezo-to-post-relaxation avalanche lead-time analysis. Old release signals are same-step effects; a new stored-potential sensor did not pass the spatially averaged nine-episode test. Event-nearest diagnostic support is not usable for prediction because it relies on future event coordinates.
* Fixed mountain target refill so `TARGET_FILL_MODE=sources` deposits refill mass at persistent source locations instead of uniformly random cells. The episode batch now uses this localized loading mode by default.
* Added causal `top_k` and `top_k_rise` sensor pooling to the lead-time analyzer. Both fail on the nine-episode localized default, so dynamic pooling alone does not recover the oracle diagnostic.
* Made `SOURCE_COUNT` explicit in the episode-batch runner and derived its end time from simulated steps, removing 34 hours of false empty padding after 3,000-step runs.
* Confirmed the corrected 64-source target baseline over nine episodes: 387 labeled rows, `182/205` positive/negative, temporal positive-rate delta `0.182`, and h6 temporal balanced accuracy `0.518750` for the simple event-list model.
* Rejected the three-episode potential-lead result: its 15--30 and 180--360 step effects did not survive nine-episode causal confirmation across 40 events.
* Added modular pre-relaxation spatial-state metrics: near-critical contact count, coherence, and weighted stress. Their encouraging three-episode screen failed nine-episode causal confirmation across 39 events, so they remain diagnostics only.
* Added opt-in delayed local failure through `DamageConfig`: near-critical cells accumulate damage, damage lowers only their local relaxation threshold, and toppling resets it. The damage-enabled nine-episode run has 387 labeled rows, `197/190` positives/negatives, drift `0.245581`, and a confirmed pre-relaxation `damage_total` lead at 5--15 steps (`AUC 0.652315`, positive in 6/9 episodes).
* Added named piezo-channel exclusions to group holdout for matched feature ablations. On the damage profile, the single-seed nine-fold Transformer screen is lower with damage channels (`0.586848`) than without (`0.599648`); do not promote them yet.
* Added 15-minute synthetic step targets sampled every 5 minutes to match the damage lead. The matched short-horizon nine-fold screen is also lower with damage (`0.529700`) than without (`0.530826`), so a generic Transformer does not yet exploit the causal state.
* Tested dedicated damage-only patch-Transformer branches at 12- and 24-minute lookback. They reach `0.500824` and `0.504695`; isolating or extending context does not recover predictive utility.
# Japan parallel data path

* Run `./scripts/backfill-japan-history.sh` and verify nonempty USGS raw and normalized outputs.
* Identify one reproducible current passive broadband ELF/VLF Japan sample and add it to `data/raw/vlf/japan/manifest.csv`; prioritize ISEE Moshiri or Kagoshima over WALDO.
* Compare Japan and Italy source coverage before any cross-region model training.
* Contact the ISEE data owners for one recent Moshiri or Kagoshima digital sample, then build the Japan VLF adapter for its native binary format.
* Keep WALDO out of the main acquisition schedule; revisit it only for a defined historical case study or optional self-supervised pretraining corpus.

* Refreshed Italy data through 2026-07-16: 67 new INGV events were pulled, one new Cumiana `last_E_VLF` image was captured, and the prospective tables now contain 279 rows with 277 mature rows. Both all-Italy (`277/0`) and central-Italy (`0/277`) remain class-blocked.
* Rebuilt the real VLF sequence and model inputs: 279 image rows and 256 anomaly windows now extend through 2026-07-16. The label-free smoke forecast remains a novelty artifact, not a seismic prediction.
* Audited the mirrored all-Italy/central-Italy label counts. Equal row counts are correct because both scopes use the same VLF anchors; the one-class labels are target saturation, not a region-filter bug. All-Italy is `277/0` at M3+ and central Italy is `0/277`; at M2.5+ central Italy is `228/49` but all-Italy remains `277/0`.
* Added `./scripts/report-italy-data-coverage.sh`. The latest report contains 4,836 INGV events, 283 Cumiana capture metadata records, 256 VLF anomaly windows, and only two weeks with both VLF and seismic observations. This is descriptive coverage evidence, not an association result.
* Added `./scripts/analyze-italy-vlf-event-association.sh`. The first refreshed permutation-controlled association remains `insufficient_controls`: three VLF-observed weeks provide only one M2.5+ event week and two controls.

## Italy coverage diagnostics

* Run `./scripts/report-italy-data-coverage.sh` after each refresh. It reports INGV event coverage, Cumiana capture coverage, label-free anomaly coverage, and descriptive weekly overlap.
* Treat anomaly/event overlap as exploratory only until enough mature windows contain both positive and negative targets.
* Replace binary all-Italy targets with fixed spatial-cell targets or count regression; use central-Italy M2.5+ only as a temporary exploratory control.

The fixed-cell implementation is available in `data/derived/multimodal/all_italy.spatial_vlf_image_windows.labeled.csv` and is prepared by `./scripts/prepare-italy-spatial-model-inputs.sh`. The current smoke artifact has 5,301 rows across 19 cells, with 812 positive, 4,451 negative, and 38 pending labels. This fixes target saturation but does not fix the short time coverage or establish predictive skill.

The first grouped-time logistic smoke baseline reached calibrated balanced accuracy `0.655320` for the all-feature ablation. The seismic-only and VLF-only ablations collapsed to balanced accuracy `0.5` under their calibrated thresholds. These figures are a single short-window diagnostic and are not evidence that either modality predicts earthquakes.

The first 19-cell leave-one-cell-out probe is stored under `data/derived/models/all_italy_spatial_cell_holdouts_v2`. Only 5 cells have positive test labels; the other 14 folds are one-class. The valid folds range from `0.146597` to `0.855263` calibrated balanced accuracy, with mean `0.333370` across all folds. This instability and class sparsity block meaningful spatial transfer evaluation.

The timestamp-permutation null control is stored under `data/derived/models/all_italy_spatial_permutation_controls`. It preserves each timestamp's complete spatial label pattern but shuffles those patterns across time. Five controls scored `0.643309`--`0.709108`, mean `0.679362`; all five matched or exceeded the real-order `0.655320`. The current multimodal score therefore has no demonstrated temporal signal.
