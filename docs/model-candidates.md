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
* `elfquake.models.window_adapter` - aggregates irregular real or synthetic event lists into regular window features.
* `elfquake.models.sequence_materializer` - materializes `time x entity x channel` sequence tables with present masks.
* `elfquake.models.tensor_spec` - CSV-to-tensor metadata spec with modality groups and generated present-mask channel names.
* `elfquake.models.tensor_materializer` - backend-neutral values, mask, and index CSV materialization from a tensor spec.
* `list-model-candidates` - writes the candidate registry JSON.
* `build-alignment-manifest` - writes a model-run manifest across tensor and sequence datasets.
* `build-aligned-window-dataset` - writes aligned regular-window model rows for synthetic or real inputs.
* `evaluate-temporal-holdout` - trains on earlier labeled rows and evaluates on later labeled rows.
* `audit-model-interfaces` - classifies event lists, image feature tables, sensor series, and summary series.
* `build-event-window-features` - writes regular event-window features from INGV-like event lists.
* `materialize-sequence-dataset` - writes sequence `values.csv`, `masks.csv`, axis files, and a manifest.
* `build-tensor-spec` - writes a tensor-spec JSON for a feature table.
* `materialize-tensor-dataset` - writes `values.csv`, `masks.csv`, `index.csv`, and `manifest.json`.

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
* `data/derived/models/mountain_256x256_seed42_10000.aligned_hourly_synthetic_windows_gt1.temporal_holdout.json`
* `data/derived/models/ingv_italy_2026-06-01_2026-06-30.aligned_real_windows.csv`

Keep model adapters behind small interfaces. Candidate selection, tensor specs, tensor materialization, training backends, and evaluation should remain separate modules so PyTorch, tree models, or event-process models can be swapped without rewriting feature generation.

See [Model Interface Shape](model-interface-shape.md) for the current adapter gaps.
