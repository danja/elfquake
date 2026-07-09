# Modeling and Evaluation Strategy

This document outlines the core modeling principles, target labeling rules, training readiness gates, evaluation metrics, and validation splits for the ELFQuake project.

---

## 1. Core Principles

*   **Baseline-First**: Begin with naive historical-rate baselines and classical ML before deploying deep neural networks.
*   **Predictive Value Proof**: Do not claim value from VLF radio or astronomical features without explicit ablation tests comparing them directly to seismic-only baselines.
*   **No Prediction Assumptions**: Treat negative results as valid research outcomes. Avoid claiming prediction capability unless it is demonstrated on held-out data.
*   **Ablation Groups**: Evaluate all models under:
    1.  Seismic-only features.
    2.  Seismic + VLF features.
    3.  Seismic + Astronomical features.
    4.  Seismic + VLF + Astronomical features.

---

## 2. Prediction Targets and Labeling Rules

We define prediction targets before choosing algorithms. The target labels are assigned dynamically using the `label-multimodal-targets` tool.

### Target Specifications
*   **Primary Real Target**: Predict whether at least one seismic event with magnitude `>= 3.0` occurs in a given region (e.g. `central_italy`) within the next `7` days.
*   **Synthetic Target (`gt0`)**: For synthetic sandpile simulations, the target is defined as a next-hour avalanche event count greater than `0`. (The `gt1` target is kept only as a sparsity diagnostic due to event sparsity).
*   **Synthetic Event-List Target**: For forecast-shape engineering, derive future event count, occurrence, max magnitude, centroid, and time-to-first-event directly from avalanche event CSVs. Keep these targets separate from piezo/VLF analogue features so the VLF path cannot define the seismic target.

### Labeling Rules
*   **Temporal Bounds**: Count seismic events in the target interval `[target_start_utc, target_end_utc)`.
*   **Geographic Bounds**: Filter event locations by the designated region (`italy_region`).
*   **Maturity (Lookahead Gate)**: Leave target rows unlabeled (marked `unlabeled_pending_future_events`) when the target window's end time `target_end_utc` is after the execution/query timestamp `--as-of`. Do not leak future data.

---

## 3. Model Training Readiness Gates

Before executing model training (especially for real datasets), we verify dataset readiness using the `summarize-model-readiness` tool:
1.  **Class Balance Check**: Verify that both positive and negative target labels exist in the training subset.
    *   *Constraint*: Real prospective training is blocked if the dataset is one-class (e.g. all positive or all negative).
2.  **Feature Group Completeness**: Check that all planned feature columns (seismic count, Kp/ap, crop VLF images) are populated and have no missing entries.

---

## 4. Evaluation and Validation Metrics

To prove the system adds value beyond simple historical baselines, models must report:

### Core Metrics
*   **Precision and Recall** (by magnitude threshold).
*   **False Positive (FPR) and False Negative (FNR) Rates**.
*   **Probability Calibration**: Group predictions into probability buckets and compare predicted vs. observed frequencies.
*   **Skill Scores**: Relative score comparisons against naive always-negative and historical-rate baselines.

### Validation Scheme
*   **Time-Based Backtesting**: Training data must strictly occur chronologically before validation data.
*   **Balanced Synthetic Engineering Splits**: Use only to test whether a synthetic target shape is learnable. A balanced split passing does not replace time-based validation.
*   **Seed Holdouts**: For synthetic datasets, evaluate models using leave-one-seed-out group holdouts.
*   **Geographic Holdouts**: Implement regional splits once spatial data density permits.
