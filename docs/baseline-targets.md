# Baseline Targets

Define simple targets before adding machine learning.

## First Target

Predict whether at least one event with magnitude `>= 3.0` occurs in Central Italy within the next `7` days.

Input table: [Baseline Input](baseline-input.md).

Smoke run: [Smoke Baseline Run](baseline-run-smoke.md).

## Baseline Model

Use a historical-rate baseline:

* train on past event frequency only
* group by region and time window
* output an event probability for each window
* avoid radio, waveform, or astronomical features

## Metrics

* precision and recall
* false positive and false negative rates
* calibration by probability bucket
* comparison against always-negative and historical-rate baselines

## Split

Use time-based validation. Training data must occur before validation data.

The current smoke dataset is only for table-shape validation. It is too small for meaningful baseline evaluation.
