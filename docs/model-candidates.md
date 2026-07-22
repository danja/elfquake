# Model Candidates

Start with models that test whether VLF and astronomy add value over seismic-only features. Do not move to deep models until time-based validation and ablations are reproducible.

For the current transfer trial, see [Feature And Training Options](feature-training-options.md). The immediate bottleneck is the sparse four-feature seismic representation and absent real VLF/astronomy history, not Transformer capacity.

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
2. Default to self-supervised real VLF pretraining while supervised real labels are one-class or sparse.
3. Use synthetic event-list heads to exercise forecast-shaped occurrence, count, magnitude, centroid, timing, rate, and spread outputs before real labels mature.
4. Use synthetic supervised pretraining as a secondary engineering check, not the default real-data path.
5. Fine-tune on real VLF-aligned rows only after real labels contain both positive and negative examples.
6. Add cross-modality attention only after unimodal and full-sequence ablations are stable.
7. Explore frequency-biased, graph, and event-process Transformers as research branches, not first production models.

Current event-list adapter: dependency-light heads with an 8-member feature-bag occurrence ensemble. It now predicts occurrence, count, max/mean magnitude, centroid, event-rate, log magnitude energy, within-horizon count bins, peak timing, duration, and spatial spread. This improves the latest scaled synthetic checks modestly, but temporal utility remains below the synthetic gate, so do not promote it to a forecast adapter yet.

Optional stronger head: a boosted-stump occurrence head is available for nonlinear tabular checks. It improved the balanced engineering split but failed the chronological split, so it remains a diagnostic branch rather than the default.

Probe harness: `probe-synthetic-event-list-models.sh` sweeps target horizon, optional burn-in trimming, feature caps, ensemble size, boosted stumps, and balanced controls, then writes a compact summary. Use this before changing the default event-list adapter so temporal improvements can be separated from balanced-split learnability.

Latest probe conclusion: one-row tabular event-list heads are not enough. The best drift-ok chronological result remains below the synthetic gate, while balanced controls are learnable. The next candidate should carry explicit temporal context, either as lagged event-list features, a sequence encoder, or an event-process head over recent avalanche/piezo history.

Lagged-context result: adding recent feature history to h6 targets improves the best drift-ok chronological score to `0.587093` with a 256-feature cap, just below the synthetic gate. This supports moving next to a regularized sequence/event-process head rather than continuing to widen tabular lag features.

Current sequence-head result: a regularized CPU GRU over grouped synthetic event-list rows is now the leading event-list candidate. The h6 lookback-12 seed-42 run passes the synthetic gate at `0.609649` calibrated balanced accuracy, but seed/config variation is still too large for promotion. Next, repeat seeds systematically and add count/location heads on the same temporal representation.

Current MLP baseline and feature check: `run-transfer-experiments.sh` accepts `FEATURE_MODE=compact|multiscale`. The multiscale mode adds causal local/neighbour counts, maximum magnitude, log energy, and recency over 1--90 day windows. On the matched three-episode transfer trial it scored `0.907098` balanced accuracy and `0.144231` default precision, below the compact baseline (`0.919624`, `0.163043`). Keep the MLP as the inexpensive baseline and keep multiscale features available for further calibration, but do not treat this first result as evidence of added value.

Current Transformer sweep: `sweep-synthetic-event-list-patch-transformer.sh` now supports multiple seeds and was run with seeds `7`, `17`, `42`, `99`, and `123`, 36 epochs, lookbacks 8/12, patch sizes 2/3, and dropout 0.1. Across 20 runs, calibrated balanced accuracy averaged `0.5620` for piezo/VLF-only, `0.4770` for direct-only, `0.5472` for direct plus piezo/VLF, and `0.5340` for full. Lookback 8, patch 3 was the best piezo/VLF configuration at `0.5912` mean. The result supports retaining piezo/VLF-only as the leading synthetic ablation, but fixed-split seed variance remains too high for model promotion.

Multi-task extension: `train-torch-patch-transformer-split-holdout` accepts repeatable `--regression-target` fields. Count and log magnitude-energy heads are trained with normalization fitted on the training rows and reported with held-out MAE. `evaluate-synthetic-event-list-patch-transformer-episodes.sh` applies these heads in nine leave-one-episode-out folds. The current mean occurrence score is `0.5508` with a `0.3788`--`0.7009` range; keep this as a regime-generalization diagnostic, not a selected model.

The matched occurrence-only control scored `0.5533` mean balanced accuracy with a `0.4091`--`0.6830` range. Auxiliary count and energy losses did not improve occurrence, so occurrence-only remains the default head while the regression outputs remain optional diagnostics.

Domain-robust feature probe: `FEATURE_MODE=relative` expresses local activity against a causal Italy-wide baseline. Its matched rolling mean balanced accuracy was `0.907935`, below multiscale `0.912232`, so relative scaling is not currently preferred. Keep it available for future domain-randomized simulation batches.

First domain-randomization stress test: `data/derived/models/domain_randomized_transformer_episode_holdout/summary.json` covers 12 complete episodes from three loading regimes. Mean calibrated balanced accuracy was `0.5096`, with profile means `0.5042` baseline, `0.4881` slow-fill, and `0.5364` fast-localized. The current model therefore fails the regime-invariance gate; improve feature normalization and target alignment before widening the domain.

The Transformer now has an opt-in `--sequence-normalization per_window` control. On the same randomized folds it scored `0.4909`, below global normalization (`0.5096`), so per-window normalization is not a default and should not be used to rescue the current model.

First stability sweep: `sweep-synthetic-event-list-sequence-head.sh` selected lookback `12`, dropout `0.1` as the best mean h6 configuration, with mean balanced accuracy `0.600459` over seeds `7`, `42`, and `99`. This should be treated as the current default occurrence candidate, but not yet the forecast adapter because only one of three seeds passed the gate.

Ensemble check: probability averaging over all three default seeds scored `0.591479`, while the best pair, seeds `42+99`, scored `0.644110`. Do not hard-code a cherry-picked seed pair as the default; use this result to motivate validation-selected ensembles or early stopping.

Validation and early-stopping check: both underperformed on the current h6 table. A 20% internal validation slice appears too small or shifted to select thresholds or stopping points reliably. Keep full-epoch lookback-12/dropout-0.1 as the leading occurrence candidate until larger synthetic batches or a better validation design are available.

## Default Self-Supervised Path

The default modeling path is now a CPU self-supervised real VLF sequence autoencoder. It learns an embedding from Cumiana spectrogram image features without earthquake target labels, so it can improve as captures accumulate even while supervised target tables are blocked.

Current implementation:

* `pretrain-sequence-autoencoder` trains a masked reconstruction model on any materialized sequence manifest.
* `pretrain-real-vlf-self-supervised.sh` runs that command on `data/derived/models/cumiana_vlf_image_sequence/manifest.json`.
* `evaluate-self-supervised-transformer` uses modality-specific patch adapters, a shared temporal Transformer, observed/corruption/padding masks, elapsed-time inputs, modality dropout, and masked reconstruction heads.
* `evaluate-self-supervised-transformer.sh` compares random initialization, synthetic pretraining, real VLF pretraining, sequential synthetic-to-real pretraining, and balanced joint pretraining over seeds `7`, `42`, and `99`.
* Static VLF image geometry is excluded from reconstruction; only signal-shape fields are targets.
* Current smoke artifact: `data/derived/models/self_supervised/real_vlf_image_autoencoder.json`.
* Current checkpoint: `data/derived/models/self_supervised/real_vlf_image_autoencoder.pt`.
* Current embeddings: `data/derived/models/self_supervised/real_vlf_image_embeddings.csv`.
* Tuned smoke run used 247 real VLF rows, 224 windows, and a chronological 179/45 train/test split. Test masked MSE was `0.835488` against a zero baseline of `1.074356`.
* `score-real-vlf-anomaly-forecast.sh` adds a second label-free layer: a real VLF descriptor autoencoder that scores reconstruction and embedding novelty by window, then emits a 7-day smoke forecast artifact from the latest VLF window.
* Current anomaly forecast: `data/derived/models/self_supervised/real_vlf_anomaly_forecast.json`.
* Current anomaly scores: `data/derived/models/self_supervised/real_vlf_anomaly_scores.csv`.
* Latest label-free smoke forecast window is `2026-07-06T06:50:50Z` to `2026-07-13T06:50:50Z`, with demo probability `0.952514` and demo predicted event `1`. This is an anomaly score, not a label-trained earthquake forecast.
* `compare-vlf-embedding-domains.sh` trains a shape-descriptor autoencoder on real VLF windows and encodes synthetic piezo/VLF windows through that same model.
* Current domain diagnostic: `data/derived/models/self_supervised/real_vlf_vs_synthetic_piezo_embedding_domain.json`.
* Tuned shape-profile diagnostic used 224 real VLF windows and 59,931 synthetic piezo/VLF windows. The synthetic centroid distance was `1.291640`; the synthetic-to-real nearest mean distance was `1.846295`.
* The diagnostic now marks the closest 25% synthetic descriptor windows as `is_synthetic_inlier` in `real_vlf_vs_synthetic_piezo_embeddings.csv`.
* Current inlier subset: 14,983 synthetic windows, nearest mean distance `1.162097`, scale mean absolute delta `0.057490`.
* `evaluate-vlf-synthetic-inlier-transfer.sh` trains a masked descriptor autoencoder only on that synthetic inlier subset, then evaluates reconstruction on held-out real Cumiana VLF descriptor windows.
* Current transfer diagnostic: `data/derived/models/self_supervised/real_vlf_synthetic_inlier_transfer.json`.
* Synthetic-inlier transfer used 14,983 synthetic training windows and 45 held-out real VLF windows. Held-out real masked reconstruction MSE was `0.688280` versus a zero baseline of `0.759011`; embedding centroid distance remained high at `4.281796`.
* `evaluate-vlf-mixed-domain-alignment.sh` trains a mixed real/synthetic descriptor autoencoder with local synthetic inliers, a CORAL embedding-alignment penalty, descriptor-gap reporting, and centroid/random/full controls.
* Current mixed-domain diagnostic: `data/derived/models/self_supervised/real_vlf_mixed_domain_alignment.json`.
* Mixed-domain local-inlier alignment used 14,983 balanced synthetic windows. Held-out real masked reconstruction MSE improved to `0.294475` versus a zero baseline of `0.588513`, and held-out real-to-synthetic embedding centroid distance improved to `1.033580`.
* Controls were close enough to keep pressure on the inlier method: centroid-inlier distance was `1.011474`, random was `1.142438`, and capped full-synthetic was worse at `1.617345`.

Current Transformer self-supervision result:

* Artifact: `data/derived/models/self_supervised_transformer/evaluation.json`.
* Split: 315 synthetic downstream training rows and 81 held-out rows; pretraining uses 2,048 capped synthetic windows and 186 real VLF training windows.
* Synthetic pretraining improves mean full-model balanced accuracy from `0.451754` to `0.488678`, but varies from `0.401163` to `0.572215` and does not pass the `0.60` synthetic gate.
* Balanced joint pretraining reaches mean `0.487352` with a narrower `0.466034` to `0.514382` range. Real-only pretraining reaches `0.470726`.
* Sequential synthetic-then-real pretraining falls back to `0.452264`, while its frozen probe is the strongest at `0.519176`; this is consistent with destructive updating during transfer.
* Synthetic reconstruction beats zero and last-patch baselines for all pretrained seeds. Real VLF reconstruction beats zero for all pretrained seeds but not the stronger last-patch baseline.
* Test-time missing-modality checks again favor piezo/VLF-only inputs over the full input. These are robustness checks, not separately retrained ablations.
* Class recall remains unstable: only isolated seeds keep both positive and negative recall above `0.40`, so mean initialization gains are not sufficient for promotion.

Transfer-preservation experiment:

* Artifact: `data/derived/models/self_supervised_transformer_transfer/evaluation.json`.
* `evaluate-self-supervised-transfer.sh` compares seven initialization strategies and trains independent full-input and piezo/VLF-only downstream models from every pretrained state.
* Stable name-derived initialization now makes shared parameters identical regardless of unused adapter inventory and preserves the caller's PyTorch RNG state.
* Under this control, random-init piezo/VLF-only is strongest: mean balanced accuracy `0.619033`, range `0.577723` to `0.663709`, with both recalls above `0.40` for all three seeds.
* Synthetic pretraining lowers piezo/VLF-only performance to `0.534272`. Real-only pretraining reaches `0.615157`, joint pretraining `0.614137`, sequential adaptation `0.542534`, and rehearsal `0.530396`; none establishes a gain over random initialization.
* Full-input heads remain weak. Synthetic pretraining is best at mean `0.505406`; early fusion of direct-avalanche and summary tokens remains detrimental.

Late gated-fusion experiment:

* Artifact: `data/derived/models/late_gated_fusion/evaluation.json`.
* The controlled random-init piezo/VLF anchor exactly matches the transfer harness at mean `0.619033`.
* Random naive late fusion averages `0.561506`; anchored full and direct-only fusion recover to `0.609241` and `0.606385`, but neither beats the anchor or improves worst-seed stability.
* Disabling the direct branch at test time raises the direct-only mean from `0.606385` to `0.611281`, so direct avalanche has not demonstrated added information.
* Synthetic-pretrained anchor and fusion variants remain weak at approximately `0.53`. Keep random-init piezo/VLF-only as the leading controlled architecture and do not promote direct or summary fusion.

Unseen-episode experiment:

* Artifact: `data/derived/models/piezo_group_holdout/evaluation.json`.
* `evaluate-piezo-group-holdout.sh` holds out each of nine complete simulation episodes and repeats each fold for model seeds `7`, `42`, and `99`.
* Mean balanced accuracy is `0.578712`, with a `0.275641` to `0.758730` range. Only 14 of 27 folds keep both recalls above `0.40`.
* Episode means range from `0.430199` to `0.711310`; model-seed means range from `0.536120` to `0.600213`. The within-episode score therefore overstates robustness.
* A fixed ensemble of all three predeclared model seeds improves mean episode score to `0.632634`, but only 6 of 9 episodes keep both recalls above `0.40`. Training-only threshold calibration beats a fixed `0.5` threshold (`0.601361`) but does not change which episodes pass.
* Episode diagnostics show sign changes in class-conditional piezo effects. Most h6 positives also occur in six-row runs while the model sees 12 minutes of context, indicating an unresolved target/sensor timescale mismatch.
* Controlled alternatives regress: h3/12-minute reaches `0.580991`, h6/60-minute `0.512103`, and h6/12-minute with spatial max/std sensor aggregates `0.544096` ensemble balanced accuracy.
* `compare-piezo-group-holdouts` applies the same `0.60` mean and 80% recall-fold gates to all reports; zero of four current variants passes.
* Causal pre-relaxation lead-time analysis is now available. The release channel is predominantly same-step; the stored-potential channel fails the spatial-average nine-episode confirmation. An event-nearest 1--5 step effect is diagnostic only because it chooses a receiver using the future avalanche location.
* Causal top-k and top-k-rise pooling also fail on the default nine-episode localized run. A 64-source three-episode potential effect and new pre-relaxation spatial contact/coherence features both failed nine-episode confirmation. Corrected duration-aligned targets now make this profile suitable for synthetic baseline tests, but not for precursor-feature training.
* Damage-enabled dynamics are the first supported synthetic precursor candidate. Pre-relaxation `damage_total` has a 5--15 step causal lead across nine episodes (AUC `0.652315`, 6/9 positive episode effects). However, the matched seed-42 nine-fold Transformer screen regresses from `0.599648` without damage channels to `0.586848` with them. Treat damage as a diagnostic until a parameter schedule or dedicated head improves the matched control.
* A minute-scale target resolves the obvious horizon mismatch: it samples every 5 minutes and labels an avalanche in the following 15 minutes. The same seed-42, 4-epoch, nine-fold check remains flat-to-worse with damage (`0.529700`) than without (`0.530826`). The next candidate should use imbalance-aware loss and a dedicated damage branch rather than adding damage channels to the generic piezo Transformer.
* Dedicated damage-only branches do not yet help: lookback 12 scores `0.500824`, while lookback 24 scores `0.504695`. A causal engineered damage head (level, rise, and rolling contrast) reaches `0.506188` with 30 minutes of history and `0.528976` with 60 minutes, but the latter passes both recall floors in only 4 of 9 folds. Do not add damage to the default model; first improve cross-episode stability of the delayed-failure dynamics, then sweep causal history lengths around 60 minutes.
* A persistence-only dynamics screen does not improve the candidate. Fast decay (`0.970`) confirms a 15--30 minute causal lead but regresses the nine-episode 60-minute head to `0.400773`; slow decay (`0.995`) has no supported lead in its initial three-episode screen. Retain the `0.985` control and vary another physical mechanism, such as reset fraction or threshold coupling, only after defining a causal-screen stopping rule.
* Reset fraction also does not yet improve the candidate. Both `0.75` and `0.98` appeared positive in three episodes, but a nine-episode confirmation of `0.75` lost the causal lead across 43 events despite stable target timing. A matched no-damage Transformer wrapper now exists, but invoke it only after a profile passes a sufficiently large causal confirmation.
* Fixed local-damage receiver fields (local mean, maximum, active fraction, and variation) fail their own nine-episode, 43-event top-k causal screen. Do not add more receiver pooling variants. The next synthetic candidate is a two-stage microdamage-to-mature-weakness mechanism, evaluated from nine episodes before any model run.
* The initial two-stage microdamage-to-mature-weakness profile also fails its predeclared nine-episode causal confirmation: `5,373` targets have stable split drift (`-0.001174`), but neither state supports a lead across 44 events. No model run is warranted. Further synthetic mechanism changes need a stronger physical rationale than another threshold or memory parameter sweep.
* Keep leave-one-episode-out as the primary synthetic stability test. Do not increase Transformer capacity or train the potential channel until an observable, non-oracle precursor statistic passes fresh causal confirmation.

Interpret the tuned domain, transfer, and mixed-alignment diagnostics as baselines to continue improving. Held-out real masked reconstruction now beats the zero baseline in the real-trained, synthetic-inlier-trained, and mixed-domain checks. The full synthetic domain is still too broad, and close centroid/random controls mean the synthetic VLF analogue is not yet a proven real VLF substitute.

Embedding-match improvement plan:

1. Train a mixed real VLF plus synthetic piezo/VLF descriptor autoencoder instead of relying on a synthetic-only latent space.
2. Add a CPU-friendly CORAL penalty that directly aligns real and synthetic embedding means/covariances.
3. Select synthetic inliers by local nearest-neighbour distance to real VLF training windows, with per-seed caps, rather than by global centroid alone.
4. Report descriptor-level shape gaps, especially persistence and differencing statistics, so simulation-side piezo/VLF tuning can target the mismatch.
5. Keep full, centroid-inlier, and random synthetic controls beside the local-inlier run.

This first mixed-domain alignment run substantially improves the old synthetic-only transfer embedding distance, but the random and centroid controls are close enough that selection quality still needs validation on future captures. Descriptor gaps now point most strongly at `global_std`, `global_robust_range`, `feature_mean_std`, `step_diff_std`, and `step_mean_std`.

Piezo/VLF transform sweep:

* `transform-piezo-signal` creates derived piezo/VLF CSVs using deterministic high-pass filtering, short envelope decay, burst contrast, near-threshold weighting, release mixing, and sensor-gain contrast.
* `sweep-piezo-vlf-alignment.sh` materializes those variants and ranks them with a short mixed-domain alignment run.
* Current sweep artifact: `data/derived/models/piezo_vlf_alignment_sweep/piezo_vlf_alignment_sweep.csv`.
* Under the fair short-run settings, transformed variants improved held-out embedding centroid distance versus current signal: `gain_burst` `1.757251`, `fast_burst` `1.763860`, `threshold_burst` `1.790333`, current signal `1.841903`.
* Reconstruction moved the other way: current signal held-out masked MSE was `0.281600`, while the best transformed variant was `0.318687`.

Interpretation: burst/high-pass transforms can move synthetic embeddings toward real VLF, but the present variants lose useful reconstruction structure. Do not promote a transformed variant as default until it improves both embedding distance and held-out real reconstruction, or until a downstream validation shows the tradeoff is useful.

Next use of these embeddings should be descriptor tuning, transfer checks across future captures, and later supervised fine-tuning once target labels contain both classes.

## Selected Deeper Model

The selected supervised deeper model remains a CPU PatchTST-style patch Transformer with explicit modality ablations, but it is no longer the default path while real labels are one-class. Use it after self-supervised VLF pretraining and after real supervised labels become usable.

Current implementation:

* `train-deep-patch-transformer.sh` builds a post-burn-in regime-balanced synthetic split and runs a deeper patch Transformer (`d_model=64`, 3 layers, 4 heads).
* `train-torch-patch-transformer-split-holdout` remains the backend command; the wrapper selects synthetic full, direct-avalanche-only, and piezo/VLF-only evaluations.
* `prepare-transformer-target-input` maps richer target tables such as `eventlist_target_occurred` into the standard `target_occurred` and `model_split` fields expected by the Transformer trainer.
* `train-synthetic-event-list-patch-transformer.sh` is the current h6 event-list Transformer pretraining smoke over the warmed nine-episode synthetic data.
* `sweep-synthetic-event-list-patch-transformer.sh` runs bounded lookback/patch/dropout checks for that h6 event-list target.
* Synthetic pretraining now writes a reusable checkpoint: `data/derived/models/deep_patch_transformer/deep_patch_transformer_synthetic.pt`.
* `train-real-deep-patch-transformer.sh` is the real fine-tune wrapper. It uses the synthetic checkpoint and exits with a blocked JSON report until labels contain both classes.
* `sequence_real_vlf_image_only` is available for real VLF sequence probes.
* Real sequence rows without `dataset_id` now use the single matching real sequence manifest, and sequence time lookup can use the nearest prior capture timestamp.

Synthetic-to-real transfer strategy:

1. Supervised synthetic pretraining on avalanche-derived labels using direct avalanche, piezo/VLF, and summary sequence manifests.
2. Keep the encoder interface modality-aware: synthetic piezo/VLF exercises the same VLF path role as real Cumiana image features, while direct avalanche remains separate from VLF.
3. When real labels contain both classes, fine-tune on `all_italy.real_vlf_aligned_windows.csv` and `central_italy.real_vlf_aligned_windows.csv` with `cumiana_vlf_image_sequence/manifest.json`.
4. Compare fine-tuned models against no-pretraining, seismic-only, VLF-only, and multimodal ablations on held-out time periods.

Event alignment should precede larger transfer models. See [Synthetic Event Alignment Strategies](event-alignment-strategies.md). The preferred sequence is catalog-level calibration, multi-resolution count/energy/occupancy targets, then a replaceable intensity-field or point-process head. Binary occurrence remains a useful reporting view but is currently too lossy and saturated at the country level.

Current deeper-model smoke result:

* synthetic artifact: `data/derived/models/deep_patch_transformer/deep_patch_transformer_synthetic.json`
* synthetic checkpoint: `data/derived/models/deep_patch_transformer/deep_patch_transformer_synthetic.pt`
* h6 event-list Transformer artifact: `data/derived/models/synthetic_event_list_patch_transformer/h6_patch_transformer.json`
* h6 event-list Transformer sweep: `data/derived/models/synthetic_event_list_patch_transformer_sweep/summary.json`
* real fine-tune artifacts: `data/derived/models/deep_patch_transformer/all_italy.real_finetune.json` and `central_italy.real_finetune.json`
* real VLF sequence probe: `data/derived/models/deep_patch_transformer/all_italy.real_vlf_sequence_probe.json`
* best current h6 event-list Transformer row: `sequence_piezo_vlf_only`, lookback `12`, patch `3`, dropout `0.1`, calibrated balanced accuracy `0.608629`
* current h6 event-list `sequence_full` calibrated balanced accuracy: `0.463892`
* real fine-tuning is blocked: all-Italy has `69` positives and `0` negatives; central Italy has `0` positives and `69` negatives

## Scaling Requirements

Before increasing model size, run `./scripts/estimate-model-scale.sh` and review [Model Scaling Requirements](model-scaling-requirements.md).

Current reading:

* real VLF-aligned supervised training is blocked because all-Italy and central-Italy each have one target class only
* the full synthetic 20000-step table has enough rows for a tiny synthetic-only patch/Transformer engineering check
* the post-burn-in balanced table is better for debugging split drift, but is below the larger-model row gate
* no real Transformer training should start until there are thousands of labeled real rows with both classes

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
* `elfquake.models.transformer_input_adapter` - maps richer target tables into the standard Transformer target/split contract.
* `elfquake.models.torch_patch_transformer` - CPU PyTorch patch Transformer evaluator for synthetic sequence engineering checks.
* `elfquake.models.torch_multimodal_data` - cadence-aware modality loading, train-only normalization, elapsed-time inputs, and chronological window references.
* `elfquake.models.torch_multimodal_encoder` - modality-specific patch adapters and the shared Transformer backbone.
* `elfquake.models.torch_ssl_pretrain` - masked-patch and modality-dropout reconstruction objectives and persistence baselines.
* `elfquake.models.torch_ssl_downstream` - frozen probes, fine-tuning, and test-time missing-modality checks.
* `elfquake.models.torch_ssl_transformer_evaluation` - seven-regime repeated-seed experiment orchestration.
* `elfquake.models.torch_late_gated_fusion` - independently encoded piezo/VLF anchor with optional gated auxiliary residuals.
* `elfquake.models.torch_late_gated_evaluation` - paired random/pretrained anchor, full-fusion, anchored-fusion, and direct-only evaluation.
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
* `estimate-model-scale` - reports larger-model gates, sequence feature counts, class balance, and CPU-only size guidance.
* `prepare-transformer-target-input` - writes a Transformer-ready CSV from richer target fields such as `eventlist_target_occurred`.
* `train-torch-patch-transformer-split-holdout` - trains a tiny CPU PyTorch patch Transformer on explicit train/test split rows.
* `train-deep-patch-transformer.sh` - runs the selected deeper patch Transformer synthetic pretrain smoke and writes real fine-tune readiness.
* `train-synthetic-event-list-patch-transformer.sh` - trains the current h6 event-list patch Transformer over warmed synthetic sequences.
* `sweep-synthetic-event-list-patch-transformer.sh` - repeats the h6 patch Transformer over bounded lookback/patch/dropout settings.
* `train-real-deep-patch-transformer.sh` - fine-tunes the patch Transformer from the synthetic checkpoint when real labels are ready; currently writes a blocked status report.
* `run-synthetic-diversity-smoke.sh` - generates extra CPU-only synthetic seeds without image/video overhead and refreshes model artifacts for diversity checks.

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

Current CPU PyTorch result on `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv`: `501` relabeled rows, `400/101` temporal split, best calibrated balanced accuracy `0.507576`. The `synthetic_vlf_only` and `vlf_only` ablations both use the synthetic piezo/VLF analogue on this table.

Current CPU PyTorch leave-one-seed-out results on the same table: best calibrated balanced accuracy `0.784169` for held-out seed `40`, `0.762832` for seed `41`, and `0.732602` for seed `42`. These results are more useful than the chronological split for checking synthetic transfer, but they remain synthetic-only.

Current CPU PyTorch sequence GRU results use materialized `synthetic_direct_avalanche`, `synthetic_piezo_vlf`, and `synthetic_summary` sequences with a `60` step lookback. Chronological balanced accuracy is flat at `0.500000`, matching the weak chronological regime split. Leave-one-seed-out calibrated balanced accuracy is `0.720690` for seed `40` with `sequence_full`, `0.821105` for seed `41` with `sequence_direct_avalanche_only`, and `0.768339` for seed `42` with `sequence_full`.

Current next model checks are comparison-driven: keep the selected patch Transformer as the deeper scaffold, rerun missing-modality and lookback checks after material changes, and keep the real VLF sequence manifest in the same shape as the synthetic piezo/VLF path.

Latest smoke artifacts:

* `data/derived/models/mountain_256x256_seeds40-42_20000.tabular_vs_sequence_model_comparison.json`
* `data/derived/models/sequence_sweep_smoke/sequence_sweep_comparison.json`
* `data/derived/models/sequence_sweep/sequence_sweep_comparison.json`
* `data/derived/models/model_family_comparison.json`
* `data/derived/models/sequence_modality_diagnostic.json`
* `data/derived/models/sequence_sweep_20epoch/default_vs_matched_sequence_diagnostic.json`
* `data/derived/models/sequence_training_seed_repeat/sequence_training_seed_selection.json`
* `data/derived/models/tiny_patch_transformer/tiny_patch_transformer_model_run_summary.json`
* `data/derived/models/missing_modality/missing_modality_seed42_summary.json`
* `data/derived/models/cumiana_vlf_image_sequence/manifest.json`
* `data/derived/models/all_italy.real_vlf_alignment_manifest.json`
* `data/derived/models/central_italy.real_vlf_alignment_manifest.json`

Full sequence sweep result, pre-relabel: the best calibrated sweep row is `0.766942` for `sequence_direct_avalanche_only` with `lookback=60`, `hidden=24`, held-out `seed42`. The overall family comparison still prefers the earlier default sequence report at `0.772558` for `sequence_piezo_vlf_only` on held-out `seed42`. Rerun these before using them for model selection.

The matched 20-epoch rerun is also pre-relabel. It kept `lookback=60`, `hidden=24`, `sequence_piezo_vlf_only` as the strongest single group-holdout row at `0.772558`, but should now be rerun before choosing a default sequence configuration.

Repeated training-seed runs are pre-relabel too. They previously suggested `sequence_full` was more robust than the best single piezo/VLF-only row, but the corrected-label sequence reports should now drive the next comparison.

The first tiny patch Transformer scaffold uses the post-burn-in regime-balanced split and CPU-only settings (`d_model=32`, 2 layers, 2 heads). Its current result is pre-relabel; keep it as an interface and scaling diagnostic until it is rerun on corrected labels and additional synthetic seeds.
