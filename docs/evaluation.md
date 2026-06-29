# Evaluation

Evaluation must show whether the system adds value beyond simple historical baselines.

## Metrics

* false positive and false negative rates
* precision, recall, and calibration
* lead time distribution
* spatial and temporal error within Italy
* performance by magnitude threshold
* comparison against naive historical-rate baselines
* comparison of seismic-only, seismic plus VLF, and full multimodal models
* ablation impact of VLF and astronomical feature groups

## Validation

Use time-based backtesting first. Add Italian regional holdouts when coverage is sufficient. Keep all test periods isolated from feature selection and model tuning.

Evaluate source groups early. A multimodal model is only useful if it improves held-out performance beyond seismic-only baselines without unacceptable false alarms.

## Reporting

Each evaluation report should include data coverage, target definition, model version, baseline comparison, limitations, and failure cases.
