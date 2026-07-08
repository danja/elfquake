# Model Targets

Current synthetic smoke target:

* artifact: `data/derived/models/mountain_256x256_seeds40-42_10000.aligned_hourly_synthetic_windows.csv`
* target: next-hour synthetic event count greater than `0`
* reason: after sparse `0.99/30` avalanche extraction, `gt1` has only `10` positives across `501` rows
* current class balance: `160` positive, `341` negative

Use `gt1` only as a sparsity diagnostic until longer or richer synthetic runs produce enough positive rows.
