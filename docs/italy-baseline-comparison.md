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

`scripts/run-matched-italy-ablation-suite.sh` now runs those contracts on the same weekly fixed-cell target and chronological split. Results are stored in `data/derived/models/matched_italy_ablation_suite/summary.json`.

| Contract | Synthetic pretraining | Balanced accuracy | Precision |
|---|---:|---:|---:|
| Full | Yes | 0.693435 | 0.315353 |
| Full | No | 0.678876 | 0.281046 |
| Seismic history | Yes | 0.677116 | 0.303279 |
| Seismic history | No | 0.687797 | 0.291667 |
| Seismic + VLF mask | Yes | 0.688466 | 0.302682 |
| Seismic + VLF mask | No | 0.685369 | 0.304348 |
| VLF mask | Yes | 0.601370 | 0.248148 |
| VLF mask | No | 0.601370 | 0.248148 |

The full model is the numerical best in this single fixed split, but its margin over seismic-history-only is small and the VLF mask is constant in the real holdout. No VLF utility is demonstrated. Repeat with multiple seeds and a holdout containing observed VLF before interpreting the result.
