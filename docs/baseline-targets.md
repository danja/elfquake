# Baseline Targets

This document defines the naive baseline targets and model metrics used as references before evaluating complex multimodal machine learning.

## 1. Primary Baseline Target

Predict whether at least one seismic event with magnitude `>= 3.0` occurs in Central Italy within the next `7` days.

*   **Input Schema**: Defined under [Multimodal Window Schema and Feature Extraction](multimodal-window-schema.md).
*   **Historical Smoke Run Logs**: Relocated to the [archive/](archive/) directory for reference (such as `archive/baseline-run-smoke.md` and `archive/baseline-input.md`).

## 2. Baseline Model Definition

We evaluate all features against a naive **Historical-Rate Baseline**:
*   Train on past event frequency only.
*   Group by region and time window.
*   Output the observed positive window frequency (probability) from training data for each validation window.
*   Avoid the use of VLF radio, space weather, or astronomical features.

## 3. Evaluation Metrics

Compare all candidate models against the historical-rate baseline using:
*   Precision and recall.
*   False positive and false negative rates.
*   Probability calibration by bucket.
*   Comparison against an always-negative baseline.

## 4. Chronological Validation Split

All baseline and candidate models must use a time-based validation scheme:
*   Training data must strictly occur before validation data.
*   Ensure there is no overlap between lookback windows or target horizons between training and test sets.

