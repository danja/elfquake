# Command Steps

Recommended command order for ELFQuake workflows. Run commands from the repository root with `PYTHONPATH=src`; use the project venv when optional dependencies are needed. Shell wrappers live under `scripts/`, for example `./scripts/run-all.sh`.

## 1. Plan Seismic Acquisition

### `plan-ingv-backfill`

Plan bounded INGV event fetches over a long date range before making network requests. Use this to split historical Italy pulls into reproducible chunks.

### `fetch-ingv-events`

Fetch raw INGV FDSN event text for the Italy bounding box and a UTC time range. Store the raw response unchanged, then normalize it in a separate step.

### `backfill-ingv-history.sh`

Plan, fetch, normalize, combine, and window bounded historical INGV event data. Use this for reproducible seismic-only real baseline smoke datasets before VLF-aligned real labels have both classes.

## 2. Acquire VLF Sources

### `fetch-vlf-cumiana`

Fetch the configured Cumiana live image endpoints once. This is the primary usable VLF capture path so far.

### `capture-vlf-cumiana-loop`

Run repeated Cumiana image captures at a configured interval. Use this under service supervision for prospective data collection.

### `refresh-prospective-labels.sh`

Fetch the latest INGV Italy events for the configured date window, normalize/combine them, relabel matured prospective VLF image rows, and refresh prospective summary/readiness reports.

### `record-vlf-abelian-cumiana`

Try to record a bounded Abelian Cumiana `vlf15` live Ogg audio chunk. The command rejects empty audio bodies, so it is still a validation path rather than a trusted source.

### `fetch-vlf-abelian-cumiana-archive`

Submit an Abelian `retrieve.php` archive request for Cumiana and store the response plus any nonempty generated download. Use only after confirming the requested timestamp has data.

### `probe-vlf-abelian-cumiana-archive`

Write a CSV source-validation report for Abelian archive requests. Use this before relying on Abelian historical data, because confirmed test requests have returned zero usable bytes.

## 3. Acquire Astronomy And Space Weather

### `fetch-astronomy`

Fetch JSON astronomy/space-weather sources from the manifest, including USNO moon phases and NOAA solar-cycle F10.7. This provides context features for multimodal windows.

### `fetch-gfz-kp-ap`

Fetch the GFZ Kp/ap historical text archive. Use this when building archive-grade geomagnetic features.

### `fetch-kyoto-dst`

Fetch a Kyoto WDC Dst monthly page, final or provisional. Normalize it before using it in model features.

### `fetch-ncei-goes-xrs`

Fetch a yearly GOES-15 XRS NetCDF archive file from NCEI. This requires NetCDF support for normalization.

### `fetch-f107-daily`

Fetch Space Weather Canada daily F10.7 text data. Use this for daily solar-radio-flux archive features.

## 4. Normalize Source Data

### `normalize-ingv-events`

Convert raw INGV text exports into the project event schema with UTC timestamps, Italy region labels, source paths, and provenance fields.

### `combine-normalized-events`

Merge normalized event CSVs, deduplicate by `event_id`, and sort by event time. Use this before feature windows or target labeling.

### `normalize-gfz-kp-ap`

Convert GFZ Kp/ap text into a clean CSV for feature aggregation. Keep the raw text capture alongside the normalized output.

### `normalize-kyoto-dst`

Convert Kyoto Dst monthly text/HTML extracts into hourly normalized rows. Preserve the source path for traceability.

### `normalize-f107-daily`

Convert Space Weather Canada daily F10.7 text into normalized daily rows. Use this in archive feature matrices.

### `normalize-goes-xrs`

Convert GOES XRS NetCDF variables into normalized time-series CSV rows. Use `--max-rows`, `--start`, and `--end` for bounded smoke runs.

## 5. Build Basic Feature Tables

### `build-vlf-features`

Summarize VLF capture metadata for one time window. This is a coarse availability and freshness feature builder.

### `build-vlf-window-features`

Aggregate VLF metadata over an existing training-window table. Use this for historical design matrices when VLF captures overlap the windows.

### `extract-vlf-image-features`

Extract cropped pixel statistics from Cumiana spectrogram images. These are the current main VLF features for prospective rows.

### `extract-vlf-audio-features`

Extract coarse audio/container features from Abelian Ogg/WAV captures. Use this only after nonempty audio captures exist.

### `build-astronomy-features`

Build astronomy context features for one time window. It reports capture provenance and slow context values such as moon phase and F10.7.

### `compare-vlf-image-features`

Compare a simulated VLF-like image with real Cumiana image features. Use this for sanity checks on synthetic VLF rendering.

### `join-vlf-image-features`

Join extracted VLF image features onto prospective or training windows by capture time. Run this after image feature extraction.

## 6. Build Multimodal Rows

### `write-multimodal-manifest-template`

Create a template manifest for manually specified multimodal windows. Use it when constructing small reproducible smoke datasets.

### `build-multimodal-smoke`

Build one multimodal row from seismic events, VLF metadata, and astronomy metadata. Use this as the smallest end-to-end feature smoke test.

### `build-multimodal-table`

Build a table of multimodal rows from a manifest. Use this when scaling the smoke-row pattern to multiple hand-defined windows.

### `build-prospective-vlf-windows`

Create VLF-anchored prospective rows from all available VLF captures. This is the clean rebuild command for prospective monitoring tables.

### `update-prospective-vlf-table`

Append new VLF-anchored prospective rows to an existing table without duplicating known window IDs. This is the command suited to timer/service operation.

### `summarize-prospective-table`

Summarize row counts, missing-source coverage, and target windows ready for labeling. Run this after prospective updates and after labeling.

### `label-multimodal-targets`

Label elapsed target windows from normalized event data. Only run after `target_end_utc` has passed, otherwise the rows remain pending.

## 7. Build Classical Training Features

### `build-seismic-training-windows`

Create seismic-only feature/target windows from normalized event lists. Use this before adding VLF and astronomy features.

### `build-design-matrix`

Join seismic training windows with archive Kp/ap and F10.7 features. This produces a simple tabular baseline matrix.

### `join-vlf-design-matrix`

Join VLF window features into an existing design matrix. Use this to compare seismic-only against seismic-plus-VLF variants.

### `build-event-window-features`

Aggregate irregular real or synthetic event lists into regular event-window features. Use this adapter before tensor or aligned-window materialization.

## 8. Check Readiness And Baselines

### `summarize-model-readiness`

Report labeled-row counts, available feature groups, missingness, and ablation feasibility. Run before any model training.

### `train-logistic-smoke`

Train a simple dependency-light logistic smoke model on a labeled design matrix. Treat it as a baseline sanity check, not evidence of predictive skill.

### `train-ablation-smoke`

Train simple smoke models over available feature-group ablations. Use this to compare seismic-only and multimodal variants on labeled rows.

### `evaluate-temporal-holdout`

Evaluate smoke models with a time-ordered train/test split. Use this before random splits because the project is time-series oriented.

### `train-torch-tabular-holdout`

Train a CPU PyTorch tabular MLP with the same time-ordered split and feature-group ablations. Use this as the first swappable neural baseline on synthetic aligned rows.

### `train-torch-tabular-group-holdout`

Train the same CPU PyTorch tabular MLP while holding out one dataset group, usually a synthetic seed. Use this to check synthetic transfer across generated runs.

### `train-torch-sequence-holdout`

Train a CPU PyTorch GRU over materialized sequence manifests with a time-ordered split. Use this to test whether sensor time structure adds value beyond tabular aggregates. Use repeated `--evaluation` options for focused runs such as `sequence_full`.

### `train-torch-sequence-group-holdout`

Train the same CPU PyTorch sequence model while holding out one dataset group. Use this for leave-one-seed-out or synthetic-regime transfer checks.

### `train-torch-patch-transformer-split-holdout`

Train a tiny CPU PyTorch patch Transformer over materialized sequence manifests using an explicit train/test split field. Use this only for synthetic larger-model interface checks until real VLF-aligned rows have both target classes.

### `prepare-transformer-target-input`

Map richer target tables into the standard Transformer target contract. Use this when a target such as `eventlist_target_occurred` should become `target_occurred` with a time-ordered `model_split`.

### `train-synthetic-event-list-patch-transformer.sh`

Prepare the h6 event-list target table and train the current CPU patch Transformer over warmed synthetic avalanche, piezo/VLF, and summary sequences with ablation reports.

### `sweep-synthetic-event-list-patch-transformer.sh`

Repeat the h6 event-list patch Transformer over bounded lookback, patch-size, and dropout settings. Use this after simulation, target, or model-interface changes.

### `diagnose-temporal-split`

Measure target balance and feature drift between temporal train/test partitions. Use this when holdout metrics look unstable or suspicious.

### `annotate-synthetic-regimes`

Add burn-in and regime identifiers to synthetic aligned rows. Use `--drop-burn-in` before regime holdout experiments when early simulation transients should not dominate the split.

### `assign-balanced-split`

Assign deterministic train/test labels within each group and target bucket. Use this for synthetic engineering checks that reduce seed/regime label drift, not for final forecasting validation.

### `evaluate-group-holdout`

Evaluate a model by holding out one group, usually a synthetic seed or dataset ID. Use this to test whether synthetic patterns generalize across runs.

### `summarize-model-run-reports`

Compact multiple evaluation JSON reports into one summary. Use this after temporal and group-holdout evaluations.

### `estimate-model-scale`

Estimate row counts, class balance, sequence feature size, memory footprint, and larger-model readiness gates. Use this before increasing GRU size or adding a Transformer.

### `estimate-model-scale.sh`

Run the default scale estimates for current synthetic, post-burn-in balanced synthetic, and real VLF-aligned datasets.

### `train-tiny-patch-transformer.sh`

Regenerate the post-burn-in regime-balanced synthetic split, train the tiny CPU patch Transformer, and write a compact model-run summary.

### `pretrain-sequence-autoencoder`

Train a CPU PyTorch masked reconstruction autoencoder over a materialized sequence manifest. Use this for label-free VLF representation learning before supervised real labels are usable.

### `pretrain-real-vlf-self-supervised.sh`

Run the default self-supervised path on the real Cumiana VLF image sequence manifest, writing a report, checkpoint, and embedding CSV under `data/derived/models/self_supervised/`.

### `evaluate-self-supervised-transformer`

Compare random, synthetic, real VLF, sequential synthetic-to-real, and balanced joint masked-patch initialization using frozen probes, full fine-tuning, reconstruction baselines, and test-time missing-modality checks.

### `evaluate-self-supervised-transformer.sh`

Prepare the h6 target split and run the seven-regime, three-seed CPU evaluation on current warmed synthetic sequences plus the real Cumiana VLF sequence. Each initialization trains separate full-input and piezo/VLF-only downstream models.

### `evaluate-self-supervised-transfer.sh`

Run the reproducible transfer-preservation configuration with six pretraining epochs, twelve supervised epochs, and a 2,048-window synthetic cap. Results are written under `data/derived/models/self_supervised_transformer_transfer/`.

### `evaluate-late-gated-fusion.sh`

Compare a piezo/VLF anchor with naive, anchored full, and anchored direct-only gated fusion under random and synthetic-pretrained initialization. The report includes gate statistics and test-time branch-disable checks.

### `evaluate-piezo-group-holdout`

Train the piezo/VLF-only CPU Transformer while holding out one complete synthetic episode at a time. Normalization is fitted only on training episodes; use this as the stricter synthetic generalization check.

### `evaluate-piezo-group-holdout.sh`

Prepare the default h6 target contract and run the nine-episode by three-model-seed leave-one-episode-out evaluation. Override `INPUT` and `ROOT` to compare other target horizons without overwriting the baseline.

### `compare-piezo-group-holdouts`

Rank group-holdout reports against explicit ensemble balanced-accuracy and per-episode recall-stability gates. This compares controlled variants without selecting from a held-out episode.

### `summarize-piezo-group-holdouts.sh`

Compare the current h6 baseline, h3 target, longer-context, and spatial-aggregation reports. The compact JSON is written to `data/derived/models/piezo_group_holdout_comparison.json`.

### `score-sequence-anomalies`

Train a label-free descriptor autoencoder on a materialized sequence manifest, score reconstruction and embedding novelty by window, and write a smoke forecast artifact for the configured horizon.

### `score-real-vlf-anomaly-forecast.sh`

Run the default real Cumiana VLF anomaly scorer and emit a 7-day label-free smoke forecast under `data/derived/models/self_supervised/`.

### `generate-trial-weekly-event-forecast`

Generate a deterministic trial event-list forecast for the next week. It combines historical INGV event rates and spatial density, real VLF context, astronomy context, and synthetic avalanche event priors into CSV rows with forecast time, latitude, longitude, magnitude proxy, probability proxy, and source contribution fields.

### `trial-weekly-event-forecast.sh`

Run the default current-data `>M2` weekly trial forecast and write `data/derived/models/trial_forecast/mag_gt2_weekly_trial_forecast.json` plus `mag_gt2_weekly_trial_events.csv`. This is an end-to-end contract smoke test, not a validated prediction.

### `generate-learned-weekly-event-forecast`

Train a small synthetic-window scorer on aligned synthetic rows, inject its latest learned score into the weekly event-list forecast contract, and write learned-scorer metadata beside the event CSV.

### `learned-weekly-event-forecast.sh`

Run the default synthetic-trained learned weekly forecast. Outputs are written under `data/derived/models/learned_forecast/`; the current learned scorer is a scaffold and must be evaluated against baselines before use.

### `build-synthetic-event-list-targets`

Build forecast-shaped synthetic targets from avalanche event CSVs. It adds future event count, occurrence, max/mean magnitude, centroid, first-event time, time-to-first-event, event-rate, log magnitude energy, within-horizon count bins, peak timing, duration, and spatial-spread fields without deriving targets from piezo/VLF channels.

### `build-synthetic-event-list-targets.sh`

Run the default event-list target build over the current seed `40`-`42`, `20000`-step synthetic aligned rows. The current default horizon is 6 rows because it gives the healthiest class balance.

### `build-synthetic-event-list-split.sh`

Assign a deterministic balanced engineering split for synthetic event-list targets using `eventlist_target_occurred`. Use this to check whether the target shape is learnable apart from temporal drift.

### `diagnose-synthetic-drift`

Report temporal/regime drift for a synthetic target table. It summarizes train/test target balance, per-seed time buckets, and the largest feature mean shifts while excluding target and diagnostic fields from model-feature interpretation.

### `diagnose-synthetic-event-list-drift.sh`

Run the default h6 event-list drift diagnostic. Outputs are written under `data/derived/models/synthetic_event_list_drift/`.

### `annotate-synthetic-episodes`

Add deterministic episode ids and row indexes to a synthetic table. These fields are for diagnostics and split construction, not predictive features.

### `annotate-synthetic-event-list-episodes.sh`

Annotate the default h6 event-list target table into 24-row episode blocks.

### `validate-synthetic-event-list-drift.sh`

Run the current drift-aware validation sequence: build h6 targets, diagnose drift, annotate episodes, create balanced and episode-balanced splits, and train temporal/balanced event-list heads.

### `train-synthetic-event-list-model`

Train dependency-light synthetic event-list heads for occurrence, count, magnitude, centroid, timing, rate, and spread. Use `--split-field model_split` with a balanced split for engineering checks, and omit it for the stricter temporal split. The default occurrence head uses deterministic feature-bag ensembles; `--occurrence-model-type boosted_stumps` enables a nonlinear diagnostic head.

### `train-synthetic-event-list-model.sh`

Run the default h6 synthetic event-list model. The wrapper defaults to an 8-member, 50% feature-bag occurrence ensemble. Override `INPUT`, `OUT`, `PREDICTIONS_OUT`, and `SPLIT_FIELD=model_split` to train on the balanced split; set `OCCURRENCE_MODEL_TYPE=boosted_stumps` for the optional nonlinear occurrence diagnostic.

### `summarize-synthetic-event-list-probes`

Summarize event-list model and drift probe JSON reports into one compact JSON/CSV table. Use this after variant sweeps so temporal, balanced, horizon, burn-in, and model-head checks can be compared without manual inspection.

### `probe-synthetic-event-list-models.sh`

Run the current synthetic event-list probe harness. It sweeps selected target horizons, optional burn-in trims, default and stronger occurrence heads, feature caps, and balanced controls, then writes `summary.json` and `summary.csv` under `data/derived/models/synthetic_event_list_probes/`.

### `build-synthetic-lagged-context`

Add previous-row synthetic feature history to an event-list target table while excluding target and diagnostic fields. Use this to test temporal context without leaking future labels.

### `probe-synthetic-event-list-lagged-context.sh`

Build the default h6 lagged-context table and train a temporal event-list model over it. Override `LAGS`, `MAX_FEATURE_COUNT`, and `OUT_DIR` to compare lag depth and feature caps.

### `train-synthetic-event-list-sequence-head`

Train a regularized CPU PyTorch GRU over grouped synthetic event-list rows. It predicts future event occurrence from current and recent feature history while excluding target and diagnostic fields.

### `train-synthetic-event-list-sequence-head.sh`

Run the default h6 event-list sequence-head probe. The current default uses `LOOKBACK_ROWS=12`, `DROPOUT=0.1`, AdamW weight decay, positive-class weighting, and gradient clipping; override `SEED`, `LOOKBACK_ROWS`, `HIDDEN_UNITS`, and `DROPOUT` for stability checks.

### `summarize-synthetic-event-list-sequence-heads`

Summarize event-list sequence-head JSON reports into JSON/CSV, including per-configuration mean, min, max, standard deviation, and synthetic-gate pass counts.

### `sweep-synthetic-event-list-sequence-head.sh`

Run the default h6 sequence-head stability sweep over seeds, lookback rows, and dropout values, then write `summary.json` and `summary.csv` under `data/derived/models/synthetic_event_list_sequence_sweep/`.

### `sweep-synthetic-event-list-sequence-validation.sh`

Run the h6 sequence-head sweep with an internal chronological validation slice used for threshold calibration. This is a diagnostic; current results underperform the train-calibrated default.

### `sweep-synthetic-event-list-sequence-early-stop.sh`

Run the h6 sequence-head sweep with an internal chronological validation slice and early stopping. This is a diagnostic; current results are unstable and should not be promoted.

### `ensemble-synthetic-event-list-sequence-heads`

Average train/test probabilities from repeated sequence-head reports, calibrate the ensemble threshold on averaged train probabilities, and evaluate the averaged test probabilities.

### `ensemble-synthetic-event-list-sequence-head.sh`

Regenerate the default lookback-12/dropout-0.1 sequence heads for configured seeds and write an averaged probability ensemble under `data/derived/models/synthetic_event_list_sequence_ensemble/`.

### `compare-weekly-forecasts`

Compare two weekly forecast JSON/CSV pairs against the staged success criteria. It reports count, probability, magnitude, spatial similarity, learned-scorer metrics, and whether the current artifact passes the scaffold and synthetic-utility gates.

### `compare-weekly-forecasts.sh`

Compare the default heuristic trial forecast against the default synthetic-trained learned forecast. Outputs are written under `data/derived/models/forecast_comparison/`.

### `trial-forecast-map.sh`

Render the current trial forecast event CSV on the offline Italy basemap. Use this for visual inspection of generated coordinates and magnitude-sized markers.

### `compare-sequence-embedding-domains`

Train a descriptor autoencoder on real VLF sequence windows, encode synthetic piezo/VLF windows through the same model, and write an embedding-domain diagnostic. This is a label-free shape comparison, not a prediction result.

### `compare-vlf-embedding-domains.sh`

Run the default real Cumiana VLF versus synthetic piezo/VLF embedding-domain diagnostic for the current seed `40`-`42`, `20000`-step synthetic manifests.

### `evaluate-synthetic-inlier-transfer`

Train a masked descriptor autoencoder on synthetic piezo/VLF windows selected as real-like inliers, then evaluate reconstruction on held-out real VLF descriptor windows.

### `evaluate-vlf-synthetic-inlier-transfer.sh`

Run the default synthetic-inlier transfer diagnostic for current Cumiana VLF captures and seed `40`-`42`, `20000`-step synthetic piezo/VLF manifests.

### `evaluate-mixed-domain-alignment`

Train a mixed real/synthetic descriptor autoencoder with local synthetic inlier selection, a CORAL embedding-alignment penalty, descriptor-gap reporting, and synthetic selection controls.

### `evaluate-vlf-mixed-domain-alignment.sh`

Run the default mixed-domain VLF alignment diagnostic for current Cumiana VLF captures and seed `40`-`42`, `20000`-step synthetic piezo/VLF manifests.

### `transform-piezo-signal`

Create a derived piezo/VLF CSV from an existing simulation CSV using deterministic high-pass, burst-shaping, near-threshold weighting, release mixing, and sensor-gain transforms.

### `sweep-piezo-vlf-alignment.sh`

Generate piezo/VLF transform variants, materialize sequence manifests, run short mixed-domain alignment diagnostics, and write a ranked CSV under `data/derived/models/piezo_vlf_alignment_sweep/`.

### `train-deep-patch-transformer.sh`

Run the selected deeper CPU patch Transformer on regime-balanced synthetic sequence data and write a reusable synthetic checkpoint.

### `train-real-deep-patch-transformer.sh`

Fine-tune the patch Transformer from the synthetic checkpoint on real VLF image sequences once real labels contain both classes. Until then it writes a blocked status report and exits without training.

### `run-synthetic-diversity-smoke.sh`

Generate extra synthetic seeds without heatmaps, video, or audio, then refresh event lists, aligned windows, tensors, and optional smoke reports for that seed set.

### `run-synthetic-episode-batch.sh`

Generate multiple shorter synthetic simulation episodes with localized sources, source-based target refill, more frequent bottom-layer removal, `3000` unrecorded warm-up steps, and sparse event extraction defaults. Use this when replacing one long drifting trajectory with a more diverse episode dataset.

### `analyze-piezo-event-lead-time.sh`

Measure whether pre-relaxation piezo features consistently change before direct avalanche events, using matched controls and local-baseline difference-in-differences. `mean`, `top_k`, and `top_k_rise` are causal candidate aggregations; `event_nearest` is an oracle diagnostic that may use a future event location.
Set `EVENT_SUFFIX` when comparing a non-default avalanche extraction, such as `_q998w120_aligned`.

### `build-synthetic-step-targets.sh`

Build minute-scale synthetic event labels from piezo and avalanche event CSVs. The damage experiment defaults to five-minute samples with a 15-minute look-ahead, matching the validated causal lead window.

### `evaluate-damage-precursor-head.sh`

Evaluate an interpretable, class-weighted damage-only short-horizon baseline across leave-one-simulation-episode-out folds. It uses only current and historical pre-relaxation damage values, with fold-local scaling and threshold calibration.

### `probe-damage-persistence.sh`

Run fresh nine-episode short- and long-memory delayed-failure profiles. It holds all non-memory simulation, extraction, target, and model settings constant, then writes causal-lead, drift, and 60-minute damage-head reports per profile.

### `probe-damage-reset.sh`

Run a three-episode causal screen comparing residual-damage and rapid-reset dynamics. Expand only a profile with a supported lead-time result.

### `sim.sh` with mature weakness

Set `DAMAGE_ENABLED=1` and `MATURE_WEAKNESS_ENABLED=1` to run the opt-in two-stage microdamage-to-mature-weakness mechanism. Keep its output separate from validated control profiles until a nine-episode causal check passes.

### `probe-mature-weakness.sh`

Run the predeclared nine-episode two-stage confirmation. It writes separate causal reports for microdamage and mature weakness, plus 15-minute target balance and drift diagnostics; it deliberately does not train a model.

### `evaluate-damage-profile-baselines.sh`

Compare generic piezo/VLF Transformer runs with and without the three damage channels on exactly the same profile, targets, episode folds, and initialization seed.

### `run-longer-synthetic-transformer-batch.sh`

Generate a larger warmed CPU-only synthetic episode batch for Transformer pretraining experiments. Run this only when the current short Transformer harness is stable and more synthetic coverage is worth the CPU time.

### `compare-model-run-summaries`

Compare compact model-run summary JSON files and optionally write a CSV view. Use this to compare tabular, sequence, sweep, and missing-modality runs.

### `compare-real-synthetic-models.sh`

Summarize the central-Italy historical seismic baseline and compare it against the default synthetic sequence and post-burn-in regime summaries. Use this for the compact real-vs-synthetic report.

### `diagnose-sequence-comparison`

Expand sequence model reports from a family comparison into per-evaluation diagnostics with epochs, lookback, hidden size, modality, split, and calibrated metrics.

### `compare-model-runs.sh`

Compare the current synthetic tabular and sequence PyTorch summaries. This is the quickest view of whether sequence inputs are adding value over tabular aggregates.

### `sweep-synthetic-sequence-model.sh`

Run a small sequence GRU lookback/model-size sweep over the current synthetic aligned dataset. Keep defaults small, then expand `LOOKBACKS`, `HIDDEN_UNITS_LIST`, and `EPOCHS` only when the smoke run is stable.

### `test-sequence-missing-modalities.sh`

Run sequence group-holdout checks with VLF/piezo-only and no-piezo inputs. Use this to exercise missing-modality behavior and ablation resilience.

### `diagnose-sequence-models.sh`

Run the default sequence diagnostic over `data/derived/models/model_family_comparison.json`. Use this before changing sequence defaults, because it makes epoch and modality differences explicit.

### `summarize-sequence-selection`

Summarize a sequence diagnostic by best single row, mean group score, worst held-out seed score, and temporal score. Use this to avoid choosing a model from one lucky group split.

### `summarize-sequence-selection.sh`

Run the default selection summary over the matched 20-epoch sequence diagnostic.

### `matched-sequence-comparison.sh`

Run a matched 20-epoch sequence comparison over lookbacks `30`, `60`, and `120` with hidden size `24`, then compare it against the existing default sequence reports.

### `repeat-sequence-training-seeds.sh`

Repeat the default sequence training with multiple PyTorch seeds and summarize stability across training seeds and held-out synthetic seeds.

### `train-sequence-full-regime.sh`

Annotate post-burn-in synthetic regimes, train only `sequence_full`, hold out each remaining seed/regime block, and write one compact robustness summary.

### `train-sequence-full-balanced.sh`

Annotate post-burn-in synthetic regimes, assign a deterministic regime-balanced split, train only `sequence_full`, and write a compact explicit-split summary.

### `train-real-tabular-model.sh`

Train a real-data tabular PyTorch smoke model only when the chosen real aligned table has both positive and negative labels. The script refuses to train while readiness is insufficient.

### `list-model-candidates`

Write the current model-candidate registry to JSON, optionally filtered by stage. Use this to track baseline, Transformer, and research candidates.

## 9. Materialize Model Interfaces

### `audit-model-interfaces`

Inspect tables and classify them as event lists, image feature tables, sensor time series, summaries, or windowed features. Use this before adding new data sources to model inputs.

### `build-tensor-spec`

Build a tensor materialization spec from a flat feature table, including feature groups, masks, target fields, and index fields.

### `materialize-tensor-dataset`

Materialize a tensor spec into values, masks, index CSVs, and a manifest. Use this for batch/window-shaped model inputs.

### `materialize-sequence-dataset`

Materialize sensor or simulation time-series CSVs into time/entity/channel values and masks. Use this for VLF-like or synthetic sequence inputs.

### `materialize-real-vlf-sequence.sh`

Materialize the current Cumiana image-feature table into the same sequence-manifest shape used by synthetic sequence models. This prepares real VLF data for later aligned multimodal training.

### `prepare-real-model-inputs.sh`

Build real prospective tensor specs, tensor manifests, alignment manifests, aligned window CSVs, and readiness reports for the current all-Italy and central-Italy VLF image tables. This is scaffold-only until each table has both target classes.

### `build-alignment-manifest`

Combine tensor and sequence manifests into a run-level alignment manifest. Use this to document modalities, time coverage, and ablation groups.

### `build-aligned-window-dataset`

Aggregate sequence and tensor inputs onto base window rows. When `--target-source-feature` is set, `--target-horizon-rows N` labels each row from the next `N` complete future rows. Use this to make aligned datasets for smoke training and future Transformer experiments.

### `combine-aligned-datasets`

Merge multiple aligned datasets and add dataset IDs. Use this for multi-seed synthetic experiments and grouped holdout tests.

## 10. Run Synthetic Simulation

### `benchmark-sandpile-sim`

Run a small CPU benchmark of the sandpile engine. Use this before large simulations to check runtime behavior on the current machine.

### `run-sandpile-sim`

Run the CPU sandpile simulation and write summary, sensor, piezo, avalanche-signal, activity, snapshot, and optional heatmap outputs. Use this as the source for synthetic seismic and VLF-like data.

### `summarize-sandpile-sim`

Summarize simulation outputs and basic consistency checks. Run after the sandpile simulation command.

### `render-sandpile-heatmap`

Render one saved sandpile snapshot to a PNG heatmap. Use this for visual sanity checking of terrain evolution.

## 11. Extract Synthetic Events

### `build-synthetic-event-list`

Convert sandpile summary/sensor outputs into a simple synthetic seismic event list. This is the older direct toppling-derived event path.

### `build-avalanche-signal-event-list`

Convert direct avalanche-signal outputs into an INGV-like synthetic event list. This is the preferred direct seismic analogue path.

### `tune-avalanche-event-extraction`

Search avalanche-event extraction thresholds, local-max windows, and optional max-event caps against a real event-series reference. Use this before locking synthetic event defaults.

### `tune-avalanche-events.sh`

Run the current refined central-Italy avalanche-event tuning grid against the extended historical INGV catalog. Use this to reproduce the sparse direct-seismic profile check.

## 12. Render And Compare Synthetic VLF Signals

### `render-piezo-spectrogram`

Render a spectrogram from synthetic piezo sensor outputs. Use this for frequency-domain sanity checks.

### `render-piezo-summary`

Render combined piezo time-series and spectrogram diagnostics. Use this to inspect whether the synthetic signal has visible dynamics.

### `render-piezo-audio`

Render synthetic piezo output as WAV audio. This is a sonification/debugging tool, not a model input by itself.

### `render-piezo-vlf-summary`

Render the current VLF-like piezo summary image using the display mapping intended to resemble Cumiana spectrograms.

### `compare-signal-shapes`

Compare real seismic/VLF series with synthetic seismic, piezo, and avalanche series in time and frequency domains. Use this to guide simulation tuning.

### `scan-piezo-sensors`

Compare individual piezo sensors against real VLF image traces and rank sensor shape fit. Use this before changing piezo receiver defaults.

## 13. Visualize Events

### `render-event-map`

Render real or synthetic event CSVs on an Italy map background with magnitude-scaled points. Use this for spatial sanity checks and demonstrations.

### `render-prediction-event-map`

Render avalanche-derived actual events and PyTorch predicted-positive target-window hits on one Italy map. Use this as a synthetic model demo; the model predicts windows, not epicentre coordinates.
