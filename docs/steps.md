# Command Steps

Recommended command order for ELFQuake workflows. Run commands from the repository root with `PYTHONPATH=src`; use the project venv when optional dependencies are needed.

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
