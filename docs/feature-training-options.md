# Feature And Training Options

The current real transfer trial is intentionally small: four seismic summaries, cell coordinates, and two missing-modality flags feed a weekly MLP. VLF and astronomy are not yet observed in the real holdout, so the current model cannot test their value. The next improvements should add information and improve evaluation discipline before increasing model size.

## Priority 1: Better Seismic Context

Build causal features at 1, 3, 7, 14, 28, and 90 days for each cell and its neighbouring cells:

* event count, magnitude exceedance counts, maximum and quantile magnitude
* log energy, energy rate, and count rate
* time since last event and inter-event gap statistics
* depth summaries and magnitude-weighted spatial centroid
* local-versus-neighbouring-cell activity ratios
* seasonal and long-term catalog-coverage indicators

Fit all normalization and clipping parameters on the training period only. Keep raw event catalogs unchanged.

## Priority 2: Targets And Metrics

Retain the binary M2.5+ weekly cell target, but add multi-task targets for event count, maximum magnitude, log energy, and spatial spread. A count or energy target uses more information than one thresholded label and supports better ranking.

Report precision, recall, PR-AUC, Brier score, calibration error, and precision at fixed alert budgets such as the top 1%, 3%, and 5% of cells. Balanced accuracy alone is not suitable for this rare-event task.

## Priority 3: Real Modality Quality

Replace the current binary missing flags with causal coverage and quality features:

* VLF capture age, number of captures, image-quality flags, anomaly score, and rolling anomaly statistics
* astronomy and space-weather availability, age, and rolling deviations from a training-period baseline
* separate masks for missing, stale, and low-quality observations

Use the same feature schema for synthetic piezo/VLF data and real Cumiana features. Do not encode missing values as ordinary zeros without a mask.

## Priority 4: Model Heads

Use a shared temporal encoder with replaceable heads:

1. regularized logistic and gradient-boosted count/ranking baselines
2. multi-task MLP for binary occurrence, count, energy, and magnitude
3. GRU or patch Transformer over multiscale causal windows
4. spatial intensity head over cells, with optional zero-inflated count likelihood

The spatial intensity or ranking head is a better fit than independently classifying every cell. Keep the direct seismic, VLF, and astronomy branches separate until ablations show stable gains.

## Training And Validation

Use chronological rolling folds and a final untouched period. Fit class weights, thresholds, calibration, and alert budgets on training data only. Compare:

* historical spatial-rate baseline
* seismic-only model
* seismic plus quality features
* seismic plus VLF
* full multimodal model
* missing-modality ablations

Use synthetic data for supervised pretraining and interface tests, but require matched-profile, leave-one-episode-out checks. Real VLF self-supervised pretraining remains the preferred label-free path while real target labels are sparse.

## Recommended Next Experiment

Implement the multiscale seismic and neighbouring-cell feature adapter, add count and energy targets, and rerun the current MLP before testing a deeper encoder. This isolates feature value from model-capacity changes and gives the later Transformer a stable input contract.
