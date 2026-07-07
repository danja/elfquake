# Model Comparison

Purpose: compact comparison of the current real seismic smoke baseline and synthetic sequence diagnostics.

## Current Artifacts

* summary JSON: `data/derived/models/real_synthetic_compact_comparison.json`
* summary CSV: `data/derived/models/real_synthetic_compact_comparison.csv`
* refresh command: `./compare-real-synthetic-models.sh`
* balanced split command: `./train-sequence-full-balanced.sh`

## Results

| run | split | test rows | test positives | best model | calibrated balanced accuracy |
| --- | --- | ---: | ---: | --- | ---: |
| central-Italy historical seismic-only | temporal | 5 | 1 | `all_features` | 0.500000 |
| synthetic sequence default | temporal | 201 | 128 | `sequence_direct_avalanche_only` | 0.500000 |
| synthetic sequence seed holdout | seed42 | 335 | 125 | `sequence_piezo_vlf_only` | 0.772558 |
| post-burn-in `sequence_full` | temporal | 161 | 104 | `sequence_full` | 0.509804 |
| post-burn-in `sequence_full` regime holdouts | 12 groups | 67 each | 9-46 each | `sequence_full` | mean 0.508413 |
| post-burn-in `sequence_full` regime-balanced | explicit | 162 | 72 | `sequence_full` | 0.650000 |

## Interpretation

The real central-Italy seismic-only baseline is now runnable but tiny, with only 25 windows and 5 test rows. The synthetic seed-holdout results remain stronger than temporal or regime holdouts, so they should be treated as synthetic-transfer diagnostics rather than evidence of real predictive skill.

The post-burn-in `sequence_full` failure appears dominated by regime shift: the temporal train positive rate is `0.398134`, while the test positive rate is `0.645963`. Top drift features include synthetic row/regime index and terrain height summaries, which means the split is still partly measuring simulation evolution rather than stable signal learning.

The regime-balanced split samples train and test rows within each seed/regime/target bucket. It gives a more stable engineering check: train positives/negatives are 288/354, test positives/negatives are 72/90, and `sequence_full` reaches calibrated balanced accuracy `0.650000`. This is useful for debugging synthetic model learning, but it is not a real forecasting validation split.

Real VLF-aligned model training remains blocked: all-Italy has 23 positive / 0 negative labeled rows, while central Italy has 0 positive / 23 negative.
