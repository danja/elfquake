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

* Learned scorer balanced accuracy exceeds `0.60` on unseen synthetic episodes; within-episode temporal splits are supporting diagnostics only.
* Positive and negative recall are both at least `0.40`; a model that predicts almost everything positive does not pass.
* At least 80% of repeated unseen-episode folds meet both recall floors, so a favorable mean cannot hide unstable episodes or initializations.
* Synthetic event-list targets have both classes, location targets, and a positive rate between `0.10` and `0.90`.
* Event-list adapters report occurrence, count, magnitude, centroid, timing, rate, and spatial-spread metrics, not only a binary score.
* Temporal train/test positive-rate drift is small enough for a meaningful chronological check; target positive-rate delta should be below `0.25` before promoting synthetic temporal scores.
* Forecast count calibration is within `25%` of the synthetic held-out weekly event rate.
* Ablations show whether direct avalanche, piezo/VLF-like, and combined features each add or remove value.

Current status: partly satisfied only as an engineering check. The original synthetic h6 table still fails with temporal positive-rate delta `0.609182`. The warmed aggressive profile with `WARMUP_STEPS=3000` now passes the drift gate on a nine-episode run with delta `0.187025` and 396 labeled rows, but temporal model utility remains weak. A nine-episode `WARMUP_STEPS=1000` run failed the same gate with delta `0.294146`.

Latest controlled within-episode check: stable name-derived initialization makes matched model variants directly comparable. Random-init piezo/VLF-only averages `0.619033` balanced accuracy over three seeds and keeps both recalls above `0.40` in every seed. Synthetic pretraining lowers it to `0.534272`; current self-supervision therefore does not add downstream value on this synthetic target.

Unseen-episode check: leave-one-episode-out evaluation over nine episodes and three model seeds averages `0.578712`, ranges from `0.275641` to `0.758730`, and only 14 of 27 folds keep both recalls above `0.40`. Stage 2 therefore fails despite the stronger within-episode mean.

Fixed-seed ensemble check: averaging the three predeclared model seeds raises unseen-episode mean balanced accuracy to `0.632634`, but only 6 of 9 episodes meet both recall floors. A fixed `0.5` threshold scores `0.601361` and passes the same 6 episodes, so threshold calibration is not the main instability.

Alignment checks: shortening the target to h3 scores `0.580991`; expanding h6 context from 12 to 60 simulation minutes scores `0.512103`; retaining simple spatial sensor aggregates scores `0.544096`. None passes the gate, and the original h6/12-minute mean-only representation remains the best tested configuration.

Late-fusion check: the controlled random-init anchored full and direct-only variants average `0.609241` and `0.606385`, below the matched `0.619033` piezo/VLF anchor. Disabling the direct branch raises its mean to `0.611281`, so direct avalanche has not demonstrated incremental value.

## Stage 3: Real Supervised Readiness

Success means real data can support supervised evaluation.

* Real VLF-aligned rows contain both positive and negative labels in at least one scope.
* Training rows occur strictly before validation rows.
* A seismic-only historical-rate baseline is computed for the same target and period.
* VLF and astronomical features have documented coverage and timestamp alignment.

Current status: blocked. Latest real aligned rows are one-class: all-Italy `149/0`, central Italy `0/149`, plus 129 future rows pending in each scope. Labels now require the source event catalog to cover the complete target horizon, so these counts no longer include catalog-gap rows.

## Stage 4: Predictive Claim Threshold

Success means the system demonstrates value on held-out real data.

* The multimodal model beats seismic-only and historical-rate baselines on held-out time periods.
* Improvements survive ablations: VLF and/or astronomical features must add value when included and lose value when removed.
* Probability calibration is acceptable by bucket, not only by rank metrics.
* Results are reproducible from recorded source pulls and command scripts.

Do not claim earthquake prediction capability before Stage 4 is satisfied.
