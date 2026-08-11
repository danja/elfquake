# Capture-Era Distribution Shift (2026-08-10)

Diagnostic of the confound raised in [Next Actions](next-actions.md) item 92:
the Cumiana VLF record contains a collector outage, and the chronological
train/test split of the fixed-cell Italy spatial baseline lands on it.

**Two results, pointing opposite ways.**

1. **The confound is real and explains the sub-chance score.** The
   chronological baseline scored `0.415351` calibrated balanced accuracy —
   below the 0.5 majority baseline. Rerun inside a single capture era it
   returns to `0.655320` (early era) and `0.575000` (late era). The sub-chance
   result was a cross-era artifact, not a property of the features or targets.
2. **Removing the confound does not rescue the model.** Within a single era,
   timestamp-permutation controls still match or beat the real chronological
   order, and dropping the cell coordinate columns collapses both eras to
   exactly `0.500000`. All apparent skill is a static per-cell base rate. There
   is no demonstrated temporal signal, and no demonstrated multimodal
   contribution.

Reproduce with `./scripts/diagnose-vlf-capture-era-shift.sh`, then
`./scripts/evaluate-italy-spatial-baseline.sh` and
`./scripts/evaluate-italy-spatial-coordinate-control.sh` against the per-era
CSVs it writes.

## The record is two dense eras, not a sparse era and a dense era

Item 92 described a "sparse early era" and a "dense era from 2026-07-29
onward". That is wrong. Segmenting the 580 window anchors at a 48-hour gap
threshold gives:

| Era | Span | Anchors | Labeled rows | Positive rate | Median step |
| --- | --- | --- | --- | --- | --- |
| `era_0` | `2026-06-28T09:57Z` – `2026-07-05T21:51Z` | 277 | 5,263 | `0.1543` | 30 min |
| `era_1` | `2026-07-10T06:05Z` (single capture) | 1 | 19 | — | — |
| `era_2` | `2026-07-15T12:45Z` (single capture) | 1 | 19 | — | — |
| `era_3` | `2026-07-28T08:51Z` – `2026-08-07T09:31Z` | 301 | 1,957 | `0.2427` | 30 min |

Both major eras run at the same nominal 30-minute cadence. What separates them
is an outage: gaps of 104.2 h, 126.7 h, and 308.1 h, with two isolated captures
inside it. The largest within-era gap is 15.1 h, an overnight collector stop.

The eras are of similar size in anchors (277 vs 301) but not in labeled rows
(5,263 vs 1,957), because the late era's recent windows are still pending their
target horizon.

## The split lands on the boundary

The grouped chronological split of the full table trains on
`2026-06-28T09:57Z` – `2026-07-29T08:53Z` and tests on `2026-07-29T09:23Z` –
`2026-07-31T09:38Z`. Training is therefore almost all `era_0` plus roughly one
day of `era_3`, and the test partition is entirely inside `era_3`. The split
boundary and the outage boundary are the same boundary.

This is why shuffling helped. The timestamp-permutation control mixes both
eras into training, which removes the shift the real chronological split
imposes. The controls were not measuring "no temporal signal" against a clean
null; they were measuring a strictly easier problem.

## The shift is in image content, not only in cadence

Item 92 asked whether the shift is instrumental — `vlf_capture_count`,
`vlf_latest_age_seconds`, and `vlf_total_bytes` are aggregation artifacts of
capture density and must shift mechanically with cadence. They do shift, but
they are not where the shift lives.

Median absolute standardized mean difference between `era_0` and `era_3`, by
feature family:

| Family | Features | Median \|d\| | Max \|d\| |
| --- | --- | --- | --- |
| **VLF signal** | 16 | **0.842** | **1.325** |
| VLF cadence-derived | 6 | 0.321 | 0.474 |
| Seismic | 2 | 0.508 | 0.588 |
| Astronomy | 2 | 0.261 | 0.521 |
| Quality flags | 5 | 0.000 | 0.000 |

Every one of the nine largest shifts is an image-content feature:

| Feature | Family | `era_0` mean | `era_3` mean | d | KS |
| --- | --- | --- | --- | --- | --- |
| `vlf_image_band_0_mean_latest` | vlf signal | 0.5977 | 0.4335 | -1.32 | 0.491 |
| `vlf_image_band_1_mean_latest` | vlf signal | 0.6384 | 0.5066 | -1.31 | 0.516 |
| `vlf_image_high_intensity_ratio_max` | vlf signal | 0.3018 | 0.1580 | -1.23 | 0.607 |
| `vlf_image_high_intensity_ratio_latest` | vlf signal | 0.2228 | 0.0894 | -1.19 | 0.455 |
| `vlf_image_hot_color_ratio_max` | vlf signal | 0.0278 | 0.0087 | -1.16 | 0.578 |
| `vlf_image_band_2_mean_latest` | vlf signal | 0.6086 | 0.4734 | -1.11 | 0.492 |
| `vlf_image_intensity_mean_avg` | vlf signal | 0.4686 | 0.3887 | -0.95 | 0.578 |
| `vlf_image_band_3_mean_latest` | vlf signal | 0.4154 | 0.2900 | -0.85 | 0.412 |
| `vlf_image_intensity_mean_latest` | vlf signal | 0.4888 | 0.3968 | -0.84 | 0.422 |
| `vlf_image_vertical_streak_count_max` | vlf signal | 106.01 | 127.83 | +0.63 | 0.381 |

The cadence-derived features shift much less (`vlf_capture_count` 37.25 → 33.08,
d `-0.34`; `vlf_total_bytes` d `-0.47`; `vlf_latest_age_seconds` d `+0.24`).

The pattern in the content features is coherent: the late era is uniformly
darker in the low bands, with far less high-intensity and hot-colour area, but
**more** vertical streaks. Bands 4 and 5 barely move at all (d `+0.03` each), so
the change is confined to the low bands. A uniform low-band suppression with
unchanged high bands and a constant image size (841.9 → 842.0 px wide) is more
consistent with a receiver gain, colour-scale, or display change across the
outage than with a change in propagation conditions. **This is not established
— it is the hypothesis the next check should test**, by comparing raw
spectrogram colour ranges either side of the outage rather than derived
features.

Two further shifts matter for interpretation:

* `astro_capture_count` goes `0.823` → `0.000`. The astronomy modality is
  simply absent from `era_3`. Any chronological ablation involving astronomy is
  training on a feature that is constant-zero in test.
* The targets shift too: positive rate `0.1543` → `0.2427`,
  `seismic_max_magnitude` `2.65` → `3.29`, `seismic_event_count` `4.94` → `6.85`.
  Real Italian seismicity was higher in the late era. This is a genuine target
  shift, not an artifact, and it moves in the direction that makes the
  cross-era transfer harder.

## Within-era results

Restricting to one era at a time, with all thresholds calibrated on training
rows only:

| Run | Train / test rows | Calibrated balanced accuracy |
| --- | --- | --- |
| Full table, chronological | 5,795 / 1,463 | `0.412604` |
| `era_0`, chronological | 4,199 / 1,064 | `0.655320` |
| `era_0`, cell coordinates removed | 4,199 / 1,064 | **`0.500000`** |
| `era_0`, permutation controls (n=5) | 4,199 / 1,064 | mean `0.650519`, range `0.603404`–`0.667172` |
| `era_3`, chronological | 1,558 / 399 | `0.575000` |
| `era_3`, cell coordinates removed | 1,558 / 399 | **`0.500000`** |
| `era_3`, permutation controls (n=5) | 1,558 / 399 | mean `0.592576`, range `0.572778`–`0.630354` |

The permutation controls are run at `EPOCHS=600` to match the real run. The
earlier comparison in next-actions item 8 ran controls at the script default of
100 epochs against a 600-epoch real run, which was not like-for-like.

Three of five controls beat the real chronological order in each era. With a
clean within-era null, the conclusion is unchanged from the confounded version:
**no demonstrated temporal signal.**

The coordinate control explains why. `target_cell_id` is already excluded as an
ID field, but `target_cell_latitude`, `target_cell_longitude`, and
`target_cell_degrees` still enter the design matrix, and they carry the largest
weights in every fitted model (`era_0`: longitude `+0.540`, latitude `+0.438`,
next-largest feature `+0.125`). Dropping them takes both eras to exactly
`0.500000`. The model is a static per-cell base-rate lookup. This is consistent
with the modality ablations, which already collapsed to `0.5`: `seismic_only`
and `vlf_only` contain no cell coordinates.

`era_0`'s `0.655320` is the same figure recorded as the original grouped-time
smoke baseline, because the original 5,301-row table was essentially `era_0`
alone. That number was never a cross-era result.

## Defect found: near-constant features were not treated as constant

While reading the fitted coefficients, `astro_noaa_solar_cycle_f107_value` —
constant at `125.69` throughout — was carrying the single largest weight
(`-0.894`) in the `era_0` model.

The shared standardizer guarded against zero variance with
`scales.append(scale if scale else 1.0)`. A column of 4,199 copies of `125.69`
has a mean of `125.68999999999998` after floating-point accumulation, so its
variance is `2.02e-28` and its scale is `1.42e-14` — nonzero, therefore truthy,
therefore used. Every residual is then divided by `1.42e-14`, and the constant
column standardizes to a constant `1.0` instead of `0.0`.

Here the effect was benign: the value was the same constant in train and test,
so the column acted as a second intercept and the score barely moved
(`0.415351` → `0.412604` on the full table; `era_0` and `era_3` unchanged). The
latent risk is not benign. Had the held-out partition carried a *different*
constant — the next monthly F10.7 value, or a quality flag that flips after a
collector change — the test rows would standardize to order `1e13` and saturate
every prediction.

Fixed in `src/elfquake/models/scaling.py` with a relative tolerance, applied to
all five copies of the pattern (`temporal_holdout`, `ablation_smoke`,
`logistic_smoke`, `torch_tabular`, `torch_sequence`). Regression cover is in
`tests/test_capture_era_shift.py`.

## What this does and does not settle

Settled:

* The sub-chance chronological score was a cross-era artifact. Item 92's
  hypothesis was correct on that point.
* The era shift is dominated by image-content features, not by cadence-derived
  aggregation artifacts. Item 92 left both mechanisms open; the content
  features move roughly 2.5x more than the cadence ones.
* Within a clean single-era null, the permutation result stands, and the
  fixed-cell spatial model's entire apparent skill is a static per-cell base
  rate.

Not settled:

* Whether the low-band suppression across the outage is instrumental or
  physical. Derived features cannot distinguish these; raw spectrogram
  comparison can.
* Whether `era_3` conclusions generalize. Its labeled window is only
  `2026-07-28` – `2026-07-31`, four days and 399 test rows.

Do not drop the cadence-derived features or reweight the eras on the strength
of this diagnostic. The finding that matters is not the era shift — it is that
the model has no temporal signal in either era, which no reweighting fixes.
