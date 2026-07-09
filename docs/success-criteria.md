# Success Criteria

ELFQuake should be judged in stages. Current outputs are engineering artifacts, not earthquake predictions.

## Stage 1: End-to-End Scaffold

Success means the pipeline is reproducible and inspectable.

* A single command emits a weekly `>M2` event-list CSV and JSON report.
* Each event row has time, latitude, longitude, magnitude proxy, probability proxy, source scores, and warning fields.
* The output can be rendered on the Italy map without manual conversion.
* Reports record source paths, class counts, model type, calibration context, and caveats.
* Missing modalities are represented by masks, quality fields, or explicit source metadata.

Current status: mostly satisfied by `trial-weekly-event-forecast.sh`, `learned-weekly-event-forecast.sh`, and `trial-forecast-map.sh`.

## Stage 2: Synthetic Model Utility

Success means synthetic training is better than trivial synthetic baselines.

* Learned scorer balanced accuracy exceeds `0.60` on held-out synthetic temporal or seed splits.
* Positive and negative recall are both at least `0.40`; a model that predicts almost everything positive does not pass.
* Synthetic event-list targets have both classes, location targets, and a positive rate between `0.10` and `0.90`.
* Event-list adapters report occurrence, count, magnitude, and centroid metrics, not only a binary score.
* Temporal train/test positive-rate drift is small enough for a meaningful chronological check; target positive-rate delta should be below `0.25` before promoting synthetic temporal scores.
* Forecast count calibration is within `25%` of the synthetic held-out weekly event rate.
* Ablations show whether direct avalanche, piezo/VLF-like, and combined features each add or remove value.

Current status: partly satisfied only as an engineering check. The original synthetic h6 table still fails with temporal positive-rate delta `0.609182`. The warmed aggressive profile with `WARMUP_STEPS=3000` now passes the drift gate on a nine-episode run with delta `0.187025` and 396 labeled rows, but temporal model utility remains weak. A nine-episode `WARMUP_STEPS=1000` run failed the same gate with delta `0.294146`.

## Stage 3: Real Supervised Readiness

Success means real data can support supervised evaluation.

* Real VLF-aligned rows contain both positive and negative labels in at least one scope.
* Training rows occur strictly before validation rows.
* A seismic-only historical-rate baseline is computed for the same target and period.
* VLF and astronomical features have documented coverage and timestamp alignment.

Current status: blocked. Latest real aligned rows are one-class: all-Italy `69/0`, central Italy `0/69`.

## Stage 4: Predictive Claim Threshold

Success means the system demonstrates value on held-out real data.

* The multimodal model beats seismic-only and historical-rate baselines on held-out time periods.
* Improvements survive ablations: VLF and/or astronomical features must add value when included and lose value when removed.
* Probability calibration is acceptable by bucket, not only by rank metrics.
* Results are reproducible from recorded source pulls and command scripts.

Do not claim earthquake prediction capability before Stage 4 is satisfied.
