# Model Candidates

Start with models that test whether VLF and astronomy add value over seismic-only features. Do not move to deep models until time-based validation and ablations are reproducible.

Reference: `docs/references/2202.07125v5.pdf`, *Transformers in Time Series: A Survey*. Relevant ideas are timestamp encoding, efficient attention for long sequences, patch tokens, cross-channel attention, seasonal/frequency bias, spatio-temporal graph hybrids, and event-process Transformers.

## Baseline Stage

Use these before any Transformer:

1. Historical-rate baseline by region and time window.
2. Regularized logistic regression on tabular window features.
3. Gradient-boosted trees on tabular multimodal features.

Current smoke commands:

* `summarize-model-readiness`
* `train-logistic-smoke`
* `train-ablation-smoke`
* `train-torch-tabular-holdout`

Treat current reports as wiring checks only; the labeled dataset is too small for model selection.

## Transformer Candidates

### 1. Temporal Fusion Transformer Style Tabular Sequence

Use when there are many labeled region-window rows.

Input: one token per time window, with seismic, VLF image, piezo/VLF summary, astronomy, and quality fields. Add timestamp embeddings for hour/day/month/lunar phase and source-coverage flags.

Why: interpretable variable selection and temporal attention fit the current design-matrix shape. This is the lowest-risk Transformer path.

Risks: still needs enough labeled windows; easy to overfit small data.

### 2. PatchTST-Style Channel-Independent Encoder

Use for dense time series once VLF image features, piezo simulation series, and geomagnetic/solar indexes are sampled at regular cadence.

Input: each modality channel is split into fixed-length temporal patches. Channels share the encoder, then a small head predicts the target window.

Why: patch tokens reduce sequence length and avoid noisy early cross-channel attention.

Risks: needs a stable regular sampling grid and careful missing-data masks.

### 3. Crossformer-Style Cross-Modality Encoder

Use after candidate 2 establishes stable unimodal signals.

Input: a 2D token layout of `time patch x modality/channel`. Use one attention stage across time and one across modality.

Why: explicitly tests whether VLF and astronomy interact with seismic history, matching the main hypothesis.

Risks: cross-channel attention can learn spurious correlations; require strict ablations and held-out time splits.

### 4. Frequency-Biased Transformer

Use for VLF and piezo-like signals where periodicity or 1/f structure matters.

Input: time patches plus FFT/wavelet summaries, or a FEDformer-like branch over selected frequency components.

Why: the survey highlights seasonal/frequency decomposition as a strong time-series inductive bias. This may suit VLF spectrogram-derived features better than raw pixels.

Risks: simulation frequency axes are analogical; real VLF sampling metadata must be correct before interpreting bands.

### 5. Spatio-Temporal Graph Transformer

Use when events and sensors are represented across Italian regions, grid cells, seismic stations, or VLF receiver paths.

Input: region or station nodes with temporal features; graph edges encode geography, tectonic adjacency, station distance, or receiver-path relevance.

Why: separates temporal attention from spatial relationships and can model Italy-wide regional dependence.

Risks: requires a defensible graph; not needed for the current Central Italy smoke rows.

### 6. Transformer Hawkes / Event-Process Model

Use for irregular seismic event lists rather than fixed windows.

Input: event time gaps, location/region marks, magnitude/depth marks, and optional preceding VLF/astronomy context tokens.

Why: earthquake catalogs are irregular event streams, and event-process Transformers model time intervals directly.

Risks: harder to evaluate and calibrate; keep fixed-window classifiers as the primary benchmark.

### 7. Self-Supervised Pretraining

Use only after there is enough real or synthetic sequence data.

Tasks:

* masked time-patch reconstruction
* next-window feature forecasting
* contrastive alignment between seismic, VLF, and astronomy windows
* synthetic sandpile pretraining followed by real-data fine-tuning

Why: labels will be scarce. Pretraining may help sequence encoders learn useful structure before supervised target training.

Risks: synthetic-to-real transfer may fail; evaluate against no-pretraining and seismic-only baselines.

## Recommended Order

1. Keep building labeled fixed-window tables.
2. Add regular-cadence feature tensors with masks.
3. Train candidate 1 only after classical baselines have enough positive and negative labels.
4. Test candidate 2 for dense VLF/piezo/astronomy channels.
5. Add candidate 3 only if unimodal sequence encoders beat tabular baselines.
6. Explore candidates 4-6 as research branches, not first production models.

## Evaluation Rule

Every candidate must be compared through the same time-based split:

* seismic only
* seismic plus VLF
* seismic plus astronomy
* seismic plus VLF and astronomy

Report calibration, false positives, false negatives, and performance against historical-rate baselines. No model should be described as predictive unless it beats these baselines on held-out data.

## Implementation Note

Current smoke trainers avoid external ML dependencies. A real Transformer implementation will likely require CPU PyTorch in the project venv; do not add GPU-only dependencies on the current system.

Current scaffold:

* `elfquake.models.candidates` - replaceable model-family registry.
* `elfquake.models.alignment_manifest` - links materialized datasets, time coverage, and ablation groups for one model run.
* `elfquake.models.aligned_windows` - aggregates window, timed tensor, and sequence inputs into one model-row CSV.
* `elfquake.models.interface_shape` - audits derived table shapes before choosing adapters or model backends.
* `elfquake.models.temporal_holdout` - dependency-free chronological train/test smoke evaluator for aligned rows.
* `elfquake.models.torch_tabular` - CPU PyTorch tabular MLP evaluator for aligned rows, missing masks, and modality ablations.
* `elfquake.models.dataset_combine` - combines aligned rows from multiple synthetic runs while preserving dataset provenance.
* `elfquake.models.report_summary` - compacts multiple evaluation reports into one comparison artifact.
* `elfquake.models.window_adapter` - aggregates irregular real or synthetic event lists into regular window features.
* `elfquake.models.sequence_materializer` - materializes `time x entity x channel` sequence tables with present masks.
* `elfquake.models.tensor_spec` - CSV-to-tensor metadata spec with modality groups and generated present-mask channel names.
* `elfquake.models.tensor_materializer` - backend-neutral values, mask, and index CSV materialization from a tensor spec.
* `elfquake.models.comparison` - compares compact tabular, sequence, sweep, and missing-modality model summaries.
* `list-model-candidates` - writes the candidate registry JSON.
* `build-alignment-manifest` - writes a model-run manifest across tensor and sequence datasets.
* `build-aligned-window-dataset` - writes aligned regular-window model rows for synthetic or real inputs.
* `evaluate-temporal-holdout` - trains on earlier labeled rows and evaluates on later labeled rows.
* `train-torch-tabular-holdout` - trains a CPU PyTorch MLP on earlier labeled rows and evaluates on later labeled rows.
* `evaluate-group-holdout` - trains on all labeled groups except one held-out group, currently useful for leave-one-seed-out synthetic checks.
* `summarize-model-run-reports` - writes a compact comparison across chronological and group-holdout reports.
* `compare-model-run-summaries` - compares one or more compact summary files and can emit JSON plus CSV.
* `audit-model-interfaces` - classifies event lists, image feature tables, sensor series, and summary series.
* `build-event-window-features` - writes regular event-window features from INGV-like event lists.
* `materialize-sequence-dataset` - writes sequence `values.csv`, `masks.csv`, axis files, and a manifest.
* `build-tensor-spec` - writes a tensor-spec JSON for a feature table.
* `materialize-tensor-dataset` - writes `values.csv`, `masks.csv`, `index.csv`, and `manifest.json`.
* `materialize-real-vlf-sequence.sh` - materializes current Cumiana VLF image features as a sequence manifest.
* `sweep-synthetic-sequence-model.sh` - runs a bounded sequence GRU hyperparameter sweep.
* `test-sequence-missing-modalities.sh` - exercises sequence training with VLF-only and no-VLF/piezo inputs.

Initial artifacts:

* `data/derived/models/model_candidates.json`
* `data/derived/models/interface_shape_audit.json`
* `data/derived/models/current_interface_alignment_manifest.json`
* `data/derived/models/cumiana_vlf_image_tensor_spec.json`
* `data/derived/models/cumiana_vlf_image_tensor/manifest.json`
* `data/derived/models/ingv_italy_2026-06-01_2026-06-30_daily_event_windows_tensor/manifest.json`
* `data/derived/models/mountain_256x256_seed42_10000_piezo_sequence/manifest.json`
* `data/derived/models/mountain_256x256_seed42_10000.aligned_synthetic_windows.csv`
* `data/derived/models/mountain_256x256_seed42_10000.aligned_synthetic_windows.ablation_smoke.json`
* `data/derived/models/mountain_256x256_seed42_10000.aligned_synthetic_windows.temporal_holdout.json`
* `data/derived/models/mountain_256x256_seeds40-42_10000.aligned_hourly_synthetic_windows.csv`
* `data/derived/models/mountain_256x256_seeds40-42_10000.aligned_hourly_synthetic_windows.temporal_holdout.json`
* `data/derived/models/mountain_256x256_seeds40-42_10000.aligned_hourly_synthetic_windows.model_run_summary.json`
* `data/derived/models/mountain_256x256_seed42_10000.aligned_hourly_synthetic_windows_gt1.temporal_holdout.json`
* `data/derived/models/mountain_256x256_seeds40-42_10000.aligned_hourly_synthetic_windows_gt1.csv`
* `data/derived/models/mountain_256x256_seeds40-42_10000.aligned_hourly_synthetic_windows_gt1.temporal_holdout.json`
* `data/derived/models/mountain_256x256_seeds40-42_10000.aligned_hourly_synthetic_windows_gt1.group_holdout_seed42.json`
* `data/derived/models/mountain_256x256_seeds40-42_10000.model_run_summary.json`
* `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.torch_tabular.json`
* `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.torch_tabular_group_seed40.json`
* `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.torch_tabular_group_seed41.json`
* `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.torch_tabular_group_seed42.json`
* `data/derived/models/ingv_italy_2026-06-01_2026-06-30.aligned_real_windows.csv`

Keep model adapters behind small interfaces. Candidate selection, tensor specs, tensor materialization, training backends, and evaluation should remain separate modules so PyTorch, tree models, or event-process models can be swapped without rewriting feature generation.

See [Model Interface Shape](model-interface-shape.md) for the current adapter gaps.

Chronological smoke evaluations use an `80/20` train/test split by default. For multi-seed synthetic rows, also run leave-one-seed-out evaluation before interpreting model behavior. After the `0.99/30` avalanche event extraction update, use the `gt0` hourly synthetic table for smoke modeling; the `gt1` target is now too sparse for useful training checks.

The model feature registry now declares a `vlf` role. For real rows this covers `vlf_metadata` and `vlf_image`; for synthetic test rows it also covers `synthetic_piezo_vlf`, so the piezo analogue can stand in for VLF data without mixing it with direct avalanche/seismic features.

Current CPU PyTorch result on `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv`: `1005` labeled rows, `804/201` temporal split, best calibrated balanced accuracy `0.543076` for `synthetic_seismic_piezo_vlf`. The `synthetic_vlf_only` and `vlf_only` ablations both use the synthetic piezo/VLF analogue on this table.

Current CPU PyTorch leave-one-seed-out results on the same table: best calibrated balanced accuracy `0.753501` for held-out seed `40`, `0.752810` for seed `41`, and `0.768286` for seed `42`. These results are more useful than the chronological split for checking synthetic transfer, but they remain synthetic-only.

First CPU PyTorch sequence GRU results use materialized `synthetic_direct_avalanche`, `synthetic_piezo_vlf`, and `synthetic_summary` sequences with a `60` step lookback. Chronological balanced accuracy is flat at `0.500000`, matching the weak chronological regime split. Leave-one-seed-out calibrated balanced accuracy is `0.712754` for seed `40` with `sequence_full`, `0.746127` for seed `41` with `sequence_piezo_vlf_only`, and `0.772558` for seed `42` with `sequence_piezo_vlf_only`.

Current next model checks are comparison-driven: compare tabular vs sequence reports, run a small sequence lookback sweep, test missing-modality behavior, and keep the real VLF sequence manifest in the same shape as the synthetic piezo/VLF path.

Latest smoke artifacts:

* `data/derived/models/mountain_256x256_seeds40-42_20000.tabular_vs_sequence_model_comparison.json`
* `data/derived/models/sequence_sweep_smoke/sequence_sweep_comparison.json`
* `data/derived/models/sequence_sweep/sequence_sweep_comparison.json`
* `data/derived/models/model_family_comparison.json`
* `data/derived/models/sequence_modality_diagnostic.json`
* `data/derived/models/sequence_sweep_20epoch/default_vs_matched_sequence_diagnostic.json`
* `data/derived/models/sequence_training_seed_repeat/sequence_training_seed_selection.json`
* `data/derived/models/missing_modality/missing_modality_seed42_summary.json`
* `data/derived/models/cumiana_vlf_image_sequence/manifest.json`
* `data/derived/models/all_italy.real_vlf_alignment_manifest.json`
* `data/derived/models/central_italy.real_vlf_alignment_manifest.json`

Full sequence sweep result: the best calibrated sweep row is `0.766942` for `sequence_direct_avalanche_only` with `lookback=60`, `hidden=24`, held-out `seed42`. The overall family comparison still prefers the earlier default sequence report at `0.772558` for `sequence_piezo_vlf_only` on held-out `seed42`. This is synthetic-only evidence and should not be interpreted as real predictive skill.

The matched 20-epoch rerun keeps `lookback=60`, `hidden=24`, `sequence_piezo_vlf_only` as the strongest single group-holdout row at `0.772558`. Mean group performance is still strongest for `sequence_full`, and temporal sequence rows remain near `0.5`, so keep treating these as synthetic-transfer diagnostics rather than stable model-selection evidence.

Repeated training-seed runs reinforce that interpretation: `sequence_piezo_vlf_only` still wins the best-single-row metric, but `sequence_full` wins mean group score and worst held-out seed score. Prefer `sequence_full` for robustness experiments, but do not make real claims until real data has both classes and passes held-out evaluation.
