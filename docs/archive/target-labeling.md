# Target Labeling

Use `label-multimodal-targets` after a target window has elapsed.

Rules:

* leave rows unlabeled when `target_end_utc` is after `--as-of`
* count events in `[target_start_utc, target_end_utc)`
* apply `target_magnitude_min`
* filter by `italy_region` when present

Current smoke row remains `unlabeled_pending_future_events` because its target window ends on `2026-07-06T10:15:00Z`.
