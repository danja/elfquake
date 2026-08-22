# Fixed-Cell Target Design (2026-08-20)

Answers [Findings Log](findings-log.md) item 103, and closes the question item
95(b) has been holding open: **can the fixed-cell target design express anything
beyond a static per-cell base rate?**

**It can, but barely.** Over the whole VLF-anchored record the current design
carries **27 label transitions** — 27 moments when some cell's target changes
between consecutive anchors. The grouped-time held-out partition contains
**seven**. The 11,989-row table is seven pieces of information about time.

Two artifacts support this, both reproducible:

* `./scripts/evaluate-italy-spatial-cell-stratified.sh` — scores balanced
  accuracy *inside* each cell, so a cell-constant predictor scores exactly
  `0.5`. Third standing control alongside the coordinate and permutation
  controls.
* `./scripts/diagnose-spatial-target-design.sh` — counts what a candidate
  design offers before any model is fitted.

## The row count was never the sample size

Anchors are 30 minutes apart. The target horizon is 7 days. Consecutive target
windows therefore overlap by `99.7%`, and a cell's label can only change when an
event enters or leaves the horizon. Rows are not observations; label transitions
are.

The cell-stratified control makes this visible directly:

| Era | Held-out rows | Held-out span | Cells with both classes | Held-out transitions |
| --- | --- | --- | --- | --- |
| `era_0` | 1,064 | 27.5 h | **0 of 19** | 0 |
| `era_3` | 1,368 | 56.5 h | 3 of 19 | 3 |

In `era_0` **no cell's held-out label varies at all**. The stratified metric
cannot be computed, and the `0.675409` pooled score reported in item 102 is
entirely the per-cell rate. In `era_3` three cells vary, each by exactly one
transition across 72 anchors:

| Cell | Held-out rows | Positives | Transitions | Stratified balanced accuracy |
| --- | --- | --- | --- | --- |
| `cell_01_37.25_15.25` | 72 | 56 | 1 | `0.500000` |
| `cell_12_44.75_7.75` | 72 | 38 | 1 | `0.738390` |
| `cell_14_44.75_10.75` | 72 | 34 | 1 | `0.575077` |

`all_features` scores `0.604489` stratified against `0.557356` pooled, and
`seismic_astronomy` scores `0.625000`. **Do not read those as skill.** They are
computed from three label changes. The `stratum_base_rate` control — the
per-cell training positive rate used directly as the score — behaves exactly as
designed: `0.677416` pooled, `0.500000` stratified. That is the number the
metric exists to neutralize, and it does.

## What the record can supply

The binding constraint is event supply, not window bookkeeping. Across the
802-anchor span (`2026-06-28` to `2026-08-17`) the INGV catalog holds **252
events at M≥2.0 and 68 at M≥2.5**. No horizon or cell size conjures more.

Full-record and held-out label transitions by design:

| Cells | M min | 1 day | 2 days | 3 days | 7 days |
| --- | --- | --- | --- | --- | --- |
| 0.75° | 2.0 | **111** / 32 | 88 / 23 | 83 / 18 | 61 / 10 |
| 0.75° | 2.5 | 37 / 11 | 37 / 9 | 35 / 7 | 30 / 9 |
| 0.75° | 3.0 | 21 / 7 | 20 / 6 | 19 / 5 | 19 / 5 |
| 1.5° | 2.0 | **94 / 27** | 70 / 19 | 65 / 14 | 40 / 6 |
| 1.5° | 2.5 | 31 / 10 | 31 / 8 | 29 / 6 | *27 / 7* |
| 1.5° | 3.0 | 17 / 6 | 16 / 5 | 15 / 4 | 16 / 4 |
| 3.0° | 2.0 | 51 / 14 | 20 / 7 | 9 / 2 | 2 / 0 |
| 3.0° | 2.5 | 15 / 4 | 18 / 3 | 17 / 2 | 9 / 3 |
| 3.0° | 3.0 | 7 / 2 | 7 / 2 | 8 / 1 | 7 / 2 |

*Italic* is the design in use. Three readings:

1. **Shortening the horizon helps at every cell size and threshold.** At 1.5°
   / M≥2.0, going 7 days → 1 day takes full-record transitions `40` → `94` and
   held-out `6` → `27`. Nothing is lost: the shorter horizon is strictly more
   informative because it stops one event painting a week of anchors positive.
2. **Lowering the magnitude threshold helps more than anything else.** At 1.5°
   / 1 day, M≥3.0 → M≥2.5 → M≥2.0 gives `17` → `31` → `94`. M≥2.0 is the
   catalog floor, so this is the ceiling.
3. **Coarser cells saturate.** 3.0° / M≥2.0 / 7 days has a positive rate of
   `0.879556` and **two** transitions in the entire record — the target
   saturation the fixed-cell decomposition was introduced to fix, reappearing
   once the cells are large enough.

## Recommended design

**1.5° cells, M≥2.0, 1-day horizon.** Against the current 1.5° / M≥2.5 /
7-day design: full-record transitions `27` → `94`, held-out `7` → `27`,
two-class cells `10` → `15` of 19, positive rate `0.185170` → `0.100865`.

0.75° / M≥2.0 / 1 day has more transitions still (`111` / `32`) but only
`23` of `60` cells ever carry both classes and the positive rate falls to
`0.035894`, which pushes the per-cell problem back toward one-class. Keep it as
the alternative to test once the record is longer.

The horizon is fixed when the prospective window table is built, not when the
targets are labeled, so this needs a separately scoped table rather than a
relabel. Do not repoint the live 30-minute job; write to a scoped path:

```sh
PYTHONPATH=src .venv/bin/python -m elfquake.cli build-prospective-vlf-windows \
  --events data/derived/ingv/events_italy_prospective.current.normalized.csv \
  --vlf-metadata-root data/raw/vlf/cumiana \
  --space-weather-root data/derived/astronomy \
  --region-id all_italy --lookback-hours 24 --horizon-days 1 \
  --target-magnitude-min 2.0 \
  --out data/derived/multimodal/all_italy.prospective_vlf_image_windows.h1m20.csv

INPUT=data/derived/multimodal/all_italy.prospective_vlf_image_windows.h1m20.csv \
OUT=data/derived/multimodal/all_italy.spatial_vlf_image_windows.h1m20.labeled.csv \
TARGET_MAGNITUDE_MIN=2.0 CELL_DEGREES=1.5 \
  ./scripts/build-italy-spatial-vlf-targets.sh
```

## What this does and does not settle

Settled:

* **The row count was never the sample size.** Every fixed-cell score reported
  so far — `0.415351`, `0.655320`, `0.575000`, `0.675409`, `0.557356` — was
  computed against a held-out partition containing single-digit label
  transitions. They are not refuted, they are uninformative, and the confidence
  interval on any of them spans chance.
* **The design is not the whole problem, but the current settings are the worst
  available.** 1.5° / M≥2.5 / 7 days is close to the bottom of the table on
  held-out transitions; three of the four horizon settings at the same cell and
  threshold beat it, and lowering the threshold to the catalog floor triples it.
* **`era_0` cannot be scored per cell at all.** Its held-out window is 27.5
  hours against a 7-day horizon, so no cell's label moves.

Not settled:

* Whether **any** design on this record can support a modality claim. Even the
  best one has 27 held-out transitions. The honest read is that the record is
  too short, and the fix is calendar time, not feature engineering. This is a
  data-volume conclusion, not an evidence-of-absence one.
* Whether per-cell rate *residuals* or a single regional target — the other two
  item-103 candidates — do better than a shorter horizon. Both remain untested;
  the horizon and threshold change is cheaper and larger.

Nothing here is evidence that VLF, astronomy, or seismic history predicts
earthquakes. It measures how much a target design could show if they did.

## The shortened design, built and evaluated (2026-08-21)

Answers [Findings Log](findings-log.md) item 104. The recommended design was
built to scoped `*.h1m20.*` paths and run through the standing controls. The
live 7-day table and the 30-minute systemd job were not repointed.

Two things came out of it. The design does what item 103 predicted. And the
first attempt at the evaluation was wrong, for a reason that had nothing to do
with target design.

Reproduce with the `elfquake-fixed-cell-evaluation` skill. This run is pinned at
`AS_OF=2026-08-21T11:22:36Z`, `CATALOG_END=2026-08-21T11:10:37Z`.

### The event catalog was four days stale

The live `elfquake-prospective.timer` fires every 30 minutes and had run four
minutes before the evaluation. It updates VLF image features and the prospective
window table — and **it does not fetch INGV events.** Only
`refresh-prospective-labels.sh`, run by hand, advances the event catalog. It had
last run on 2026-08-17.

So anchors kept accumulating for four days against a frozen catalog, and
`build-italy-spatial-vlf-targets.sh` declared them mature anyway: its
`CATALOG_END` defaulted to wall-clock now, which asserts coverage the catalog
did not have. **22 real events between 2026-08-17 and 2026-08-21 were labeled as
non-events**, all of them inside the held-out partition.

Refreshing changed the answer materially:

| `era_3` | Stale catalog | Refreshed |
| --- | --- | --- |
| Events in catalog | 397 | 419 |
| Held-out label transitions | 10 | **19** |
| Scoreable cells | 6 of 19 | **8** of 19 |
| `stratum_base_rate` pooled | `0.646788` | `0.534581` |
| `seismic_only` stratified | `0.652961` | `0.552836` |
| `all_features` stratified | `0.454456` | `0.404832` |

The stale run's `0.652961` was the largest apparent VLF-era result in the
project and it was an artifact of missing events. `CATALOG_END` now defaults to
the catalog's own `max(ingested_at_utc)` instead of wall-clock now, so the guard
can no longer certify coverage that does not exist.

### What the rebuild produced

| | 7-day / M≥2.5 | 1-day / M≥2.0 |
| --- | --- | --- |
| Prospective anchors | 802 | 816 |
| Spatial rows | 11,020 | 15,504 |
| Labeled rows | 7,258 | 15,181 |
| Positive rate | `0.185170` | `0.100718` |
| Full-record transitions | 27 | **104** |
| Whole-record held-out transitions | 7 | **31** |
| Cells ever two-class | 10 of 19 | **15 of 19** |

The 7-day column is the item-103 sweep, measured before the catalog
refresh; the 1-day column is this run. The comparison is indicative rather
than exact, and it understates the 1-day design if anything.

A 1-day horizon also matures far more of the record — 98% of rows labeled
against 66% — because a row becomes scoreable a day after its anchor rather
than a week after.

### Where it still fails

Item 96 forbids pooling `era_0` and `era_3`: the collector outage put an
`11.58 dB` palette/gain shift between them. Every run therefore happens inside
one era, and **the 31 held-out transitions are a whole-record figure that does
not survive the era split**:

| Era | Anchors | Labeled rows | Held-out rows | Full-record transitions | Held-out transitions | Two-class cells held out |
| --- | --- | --- | --- | --- | --- | --- |
| `era_0` | 277 | 5,263 | 1,064 | 16 | **1** | 1 of 19 |
| `era_3` | 537 | 9,880 | 1,976 | 74 | **19** | 8 of 19 |

Twenty transitions across both eras, against 3,040 held-out rows. The era
restriction and the transition budget are in direct conflict: the split that
makes the VLF features comparable is the same split that removes the label
variation needed to test them. That conflict applies to every design in the
item-103 sweep, not only this one, and it was invisible while the whole-record
figure was the only one quoted.

### The scores, each with its evidence count

`era_3`, grouped-time split, 600 epochs, stratified on `target_cell_id`, all on
the same **19** held-out transitions across 8 scoreable cells:

| Ablation | Pooled | Stratified |
| --- | --- | --- |
| `seismic_only` | `0.525464` | `0.552836` |
| `seismic_vlf` | `0.515191` | `0.539445` |
| `stratum_base_rate` control | `0.534581` | `0.500000` |
| `vlf_only` | `0.506800` | `0.532717` |
| `seismic_astronomy` | `0.500000` | `0.500000` |
| `full_multimodal` | `0.489631` | `0.431988` |
| `all_features` | `0.455515` | `0.404832` |

The spread between the best and worst ablation is `0.15`. `era_0` is worse
still: one scoreable cell, one transition, everything between `0.442308` and
`0.576923`. There is nothing to read there at all.

The `stratum_base_rate` control lands on exactly `0.500000` stratified in both
eras, as it must. The metric works; the problem is what it is being applied to.

Breaking `era_3`'s best ablation down by cell shows why:

| Cell | Held-out rows | Positives | **Transitions** | `seismic_only` | `all_features` |
| --- | --- | --- | --- | --- | --- |
| `cell_00_37.25_13.75` | 104 | 21 | 2 | `0.300344` | `0.307229` |
| `cell_01_37.25_15.25` | 104 | 25 | 2 | `0.854430` | `0.416456` |
| `cell_08_41.75_15.25` | 104 | 32 | 3 | `0.324653` | `0.227431` |
| `cell_09_43.25_10.75` | 104 | 3 | **1** | `0.777228` | `0.178218` |
| `cell_10_43.25_12.25` | 104 | 58 | 5 | `0.601949` | `0.559970` |
| `cell_11_43.25_13.75` | 104 | 31 | 2 | `0.791648` | `0.608705` |
| `cell_14_44.75_10.75` | 104 | 39 | 2 | `0.397436` | `0.707692` |
| `cell_17_46.25_10.75` | 104 | 16 | 2 | `0.375000` | `0.232955` |

Every cell rests on one to five label changes. `cell_09` holds three positive
anchors — one event seen three times — and produces `0.777228` for one ablation
and `0.178218` for another. The `0.30`–`0.85` spread across cells is what one or
two coin flips look like, and the ablation means are averages of coin flips.

### Reporting change

Because `0.652961` looked like a result until its transition count and its
catalog freshness were checked, neither is optional now. `_stratified_metrics`
emits `label_transitions` per stratum and in the summary, and the CLI prints it
with every score:

```
held-out label transitions: 19
seismic_only: 0.525464 plain, 0.552836 stratified (8/19 strata, 19 transitions)
```

Item 104(c) asked for the count beside every score. Putting it in the evaluator
rather than in a separate diagnostic means a future run cannot omit it.

### The permutation control is not a null at all

Shuffling target timestamps destroys the temporal autocorrelation that makes
consecutive rows near-copies. It therefore *manufactures* label variation, and
the transition count now makes the scale of that visible for the first time:

`all_features`, stratified on `target_cell_id`, 600 epochs both sides:

| Run | Pooled | Stratified | Strata scored | **Transitions** |
| --- | --- | --- | --- | --- |
| `era_0` real order | `0.392915` | `0.576923` | 1 of 19 | **1** |
| `era_0` seed 101 | `0.544058` | `0.478791` | 7 of 19 | 111 |
| `era_0` seed 202 | `0.601291` | `0.510036` | 7 of 19 | 94 |
| `era_0` seed 303 | `0.603008` | `0.524254` | 7 of 19 | 102 |
| `era_0` seed 404 | `0.517581` | `0.510413` | 7 of 19 | 98 |
| `era_0` seed 505 | `0.563477` | `0.460345` | 7 of 19 | 119 |
| `era_3` real order | `0.455515` | `0.404832` | 8 of 19 | **19** |
| `era_3` seed 101 | `0.542510` | `0.533923` | 12 of 19 | 322 |
| `era_3` seed 202 | `0.474119` | `0.504918` | 12 of 19 | 342 |
| `era_3` seed 303 | `0.484765` | `0.500338` | 12 of 19 | 336 |
| `era_3` seed 404 | `0.500000` | `0.500000` | 12 of 19 | 315 |
| `era_3` seed 505 | `0.525458` | `0.506722` | 12 of 19 | 317 |

The shuffled controls carry **17 times** the evidence of the run they are meant
to null, and roughly a hundred times in `era_0`. They are not the same task made
harder; they are a different, far better-resourced task. Comparing them with the
real run tells you nothing about temporal signal.

This retires the reading in item 8 — "all five shuffled controls beat real
order, so destroying temporal structure makes the task easier" — as
uninterpretable. Both the era-leak mechanism diagnosed in item 93 and this
evidence-count asymmetry are present, and neither can be separated from the
other at this record length.

It also, incidentally, validates the metric. Given 315–342 transitions the
shuffled controls land on `0.500338`, `0.504918`, `0.506722`, `0.533923`, and
exactly `0.500000` stratified — a proper null converging on chance, which is what
the cell-stratified metric is supposed to do when no signal is present. The
metric is well-behaved. The real runs simply do not have the evidence to move it.

A valid null must preserve each cell's sequence length, positive count, and
block structure, so the control and the run are scored on comparable evidence.
That is now `./scripts/evaluate-italy-spatial-shift-controls.sh`, built for item
105(d); see [Within-Cell Null Control](shift-control.md).

### Conclusion for item 104(d)

The blocker is calendar time.

Two target designs and a corrected catalog have now been tried on this record,
and the constraint has not moved: 419 Italian events at M≥2.0 across 81 days,
seen through 19 cells, split by an unavoidable era boundary, produce **twenty**
independent held-out label changes. No horizon, cell size, or magnitude
threshold in the item-103 sweep gets past that, and no feature family can be
tested against it. Adding a fifth modality would produce another table of
numbers between `0.3` and `0.85` computed from one or two coin flips.

This is a statement about data volume. It is **not** evidence that VLF,
astronomy, or seismic history lacks predictive value — that question remains
untested, and stays untested until the record is longer and unbroken by a gain
change.

What is worth doing meanwhile, in rough order of leverage:

* **Make the live collector refresh INGV.** The 30-minute service updates VLF
  and nothing else, so the catalog is stale by however long it has been since a
  human ran the refresh script. This is a live-system defect, not an analysis
  one, and it silently corrupted the first version of this evaluation.
* **Finish item 97** (palette-inverted absolute-dB VLF features). If absolute-dB
  features remove the era discontinuity, the item-96 pooling restriction lifts
  and both eras become one record — roughly doubling the usable transition count
  at zero cost in calendar time. That makes it the highest-leverage open item,
  not a feature-engineering detour.
* Build a within-cell permutation control, so the null and the run are scored on
  the same number of strata.
* Keep collecting. `era_3` is the first uninterrupted stretch and its transition
  count grows roughly linearly with span.
* Leave the per-cell rate residual and single-regional-target designs from item
  103 untested until there is enough label variation to tell them apart.
