# Italy Baseline Comparison

The two current Italy scores must not be compared as if they measured the same task.

| Property | Grouped spatial baseline | Real transfer trial |
|---|---|---|
| Input | VLF-anchored multimodal windows | Event catalog plus synthetic event catalogs |
| Temporal unit | Existing source-window target, effectively one-day target intervals | Weekly samples with a 7-day forecast horizon |
| Spatial unit | 1.5-degree cells expanded from each source window | Fixed 1.5-degree cells across Italy |
| Magnitude threshold | `>=2.5` | `>=2.5` |
| Split | 80/20 grouped by source-window time | 80/20 chronological weeks |
| Model path | Temporal holdout feature baseline | CPU PyTorch MLP with synthetic pretraining and real fine-tuning |
| VLF status | Image features present; real image sequence is missing in the current aligned input | Explicit missing VLF mask |
| Latest balanced accuracy | `0.655320` | `0.693435` |

The spatial baseline used 5,301 rows, with 1,064 held out. Its positive precision and recall were `0.259861` and `0.666667`. The transfer trial used 2,489 real samples across 104 training weeks and 27 test weeks; its precision and recall were `0.315353` and `0.783505`.

The transfer score is not evidence that synthetic pretraining, VLF, or astronomy improved prediction. The experiments differ in temporal horizon, source-window construction, feature availability, and model family. The next fair comparison should build both models from one weekly fixed-cell table, use identical chronological folds and threshold calibration, and run seismic-history-only, VLF-mask-only, full multimodal, and synthetic-pretraining ablations.

## Matched Weekly Ablation

`scripts/run-matched-italy-ablation-suite.sh` runs those contracts on the same weekly fixed-cell target and chronological split. The current suite covers seeds `42`, `99`, and `123`; results are stored in `data/derived/models/matched_italy_ablation_suite/summary.json`.

| Contract | Synthetic pretraining | BA mean (min--max) | Precision mean |
|---|---:|---:|---:|
| Full | Yes | 0.690230 (0.684675--0.693435) | 0.311912 |
| Full | No | 0.671673 (0.664292--0.678876) | 0.272594 |
| Seismic history | Yes | 0.685992 (0.677116--0.700994) | 0.309252 |
| Seismic history | No | 0.679991 (0.669905--0.687797) | 0.297930 |
| Seismic + VLF mask | Yes | 0.689776 (0.688466--0.692233) | 0.308979 |
| Seismic + VLF mask | No | 0.682618 (0.680214--0.685369) | 0.304019 |
| VLF mask | Yes | 0.601370 (0.601370--0.601370) | 0.248148 |
| VLF mask | No | 0.601593 (0.601370--0.602040) | 0.250480 |

Synthetic pretraining improves the full model by about 0.019 balanced accuracy on average, but the full and seismic-history results remain close. The VLF mask is constant in the real holdout, so this suite cannot measure VLF utility. A real VLF test period with observed variation is required before attributing any gain to that modality.
