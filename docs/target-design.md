# Fixed-Cell Target Design (2026-08-20)

Answers [Next Actions](next-actions.md) item 103, and closes the question item
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
