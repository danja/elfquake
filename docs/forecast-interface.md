# Forecast Interface

The trial forecast interface is the downstream contract for weekly event-list outputs. The current scorer is weak and heuristic; future learned scorers should preserve this shape.

## Inputs

* Real seismic history: normalized INGV event CSVs.
* Real VLF context: aligned VLF image windows, image sequences, and anomaly/self-supervised reports.
* Astronomy context: normalized or raw captured solar, geomagnetic, and lunar context.
* Synthetic artifacts: avalanche-derived seismic events and piezo/VLF-like sequences.
* Missing data: every model input path should expose a mask or quality field rather than silently filling unknown values.

## Event Output

Required CSV fields:

* `prediction_id`
* `forecast_time_utc`
* `latitude`
* `longitude`
* `magnitude_proxy`
* `probability_proxy`
* `expected_week_count`
* `real_spatial_weight`
* `synthetic_spatial_weight`
* `vlf_context_score`
* `astronomy_context_score`
* `synthetic_context_score`
* `warning`

Required JSON report fields:

* `schema`
* `status`
* `warning`
* `forecast_start_utc`
* `forecast_end_utc`
* `magnitude_condition`
* `predicted_event_count`
* `uncapped_expected_event_count`
* `sources`
* `model`
* `events_out`

## Calibration

Weekly count calibration should start from historical INGV `>M2` rates. Learned models may adjust spatial and count scores, but reports must keep historical-rate context so every run can be compared against a simple seismic baseline.

## Validation Rule

Do not present any output as earthquake prediction capability until it beats reproducible seismic-only baselines on held-out time periods and passes modality ablation checks.
