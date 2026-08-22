# Within-Cell Null Control (2026-08-21)

Answers [Findings Log](findings-log.md) item 105(d), and replaces the
timestamp-shuffle control that items 8 and 93 were read from.

Reproduce with `./scripts/evaluate-italy-spatial-shift-controls.sh`.

## The shuffle was never a null

`permute-spatial-targets` shuffles whole time slices: it keeps each anchor's
19-cell label vector intact and reassigns it to a random anchor. That looks like
a temporal null, and against independent rows it would be one.

The rows are not independent. Anchors are 30 minutes apart and the target
horizon is 1–7 days, so consecutive target windows overlap by more than 99% and
each cell's label sequence is a handful of long runs. Shuffling the order of
those rows converts a few long runs into many short ones, and every run boundary
is a label transition — the quantity [Target Design](target-design.md) shows is
the real sample size.

Measured on the item-104 tables:

| | Real order | Shuffled controls |
| --- | --- | --- |
| `era_0` held-out transitions | **1** | 94, 98, 102, 111, 119 |
| `era_3` held-out transitions | **19** | 315, 317, 322, 336, 342 |

The control carried up to a hundred times the evidence of the run it was meant
to null. It was not the same task made harder; it was a different, far
better-resourced task, and no comparison between the two means anything.

This is why item 8's reading — "all five controls beat real order, so destroying
temporal structure makes the task easier, which is the opposite of a predictive
signal" — does not survive. The controls did not beat real order by being
easier to fit. They beat it by being scored on more data.

## A circular shift is the null the shuffle was meant to be

`shift-spatial-targets` moves the whole labeled matrix in time by a single
offset, wrapping at the end. Within each cell that preserves:

* the sequence length, exactly;
* the positive count, exactly;
* the run-length structure, and therefore the transition count, apart from the
  one wrap seam;
* the spatial pattern at each anchor, since every cell moves together.

What it destroys is the alignment between features and labels, which is the only
place a temporal signal could live. Unlabeled rows keep their empty labels and
do not enter the shift — the evaluator drops them anyway, and moving them would
change which anchors are scoreable and reintroduce the asymmetry the control
exists to remove.

The offset is drawn away from both ends (`--min-shift-fraction`, default `0.1`)
so a control cannot come back as a near-copy of the real ordering.

### The preservation check

Every run prints it, and it is the check that the control is usable:

```
shift: 349 anchors across 19 cells
transitions: 74 before, 76 after
```

On `era_3`, per cell:

| Cell | Length | Positives before → after | Transitions before → after |
| --- | --- | --- | --- |
| `cell_09_43.25_10.75` | 520 | 226 → 226 | 16 → 16 |
| `cell_11_43.25_13.75` | 520 | 257 → 257 | 10 → 10 |
| `cell_14_44.75_10.75` | 520 | 156 → 156 | 9 → 10 |
| `cell_10_43.25_12.25` | 520 | 104 → 104 | 7 → 8 |
| `cell_07_41.75_13.75` | 520 | 134 → 134 | 8 → 8 |

Lengths and positive counts are identical everywhere. Two cells gain one
transition each, where the wrap seam happens to fall inside a run. Total
`74 → 76`, against the shuffle's `74 → 322`.

A before/after difference of more than two or three means the shift is not
preserving structure and the control should not be used.

## What the valid null shows on `era_0`

`era_0`, `all_features`, 600 epochs both sides, stratified on `target_cell_id`:

| Run | Pooled | Stratified | Strata scored | Transitions |
| --- | --- | --- | --- | --- |
| real order | `0.392915` | `0.576923` | 1 of 19 | **1** |
| shift 101 | `0.429952` | `0.500000` | 4 of 19 | 6 |
| shift 202 | `0.515816` | `0.611111` | 4 of 19 | 5 |
| shift 303 | `0.494048` | `0.488990` | 3 of 19 | 4 |
| shift 404 | `0.624008` | `0.525442` | 3 of 19 | 4 |
| shift 505 | `0.554688` | `0.449642` | 2 of 19 | 3 |

The evidence gap is closed to within a factor of a few: 3–6 transitions against
the real run's 1, where the shuffle gave 94–119. The controls behave like a
null should, landing on a stratified mean of `0.515` across a `0.450`–`0.611`
band centred on chance.

**A residual asymmetry remains, and it is not the shift misbehaving.**
`era_0`'s held-out window is genuinely unrepresentative of `era_0`: it holds
**6.2% of the era's transitions in 20.2% of its anchors**. Any control that
draws its held-out labels from elsewhere in the record lands on a busier
stretch. `era_3` is closer to fair at `25.7%` of transitions in `20%` of
anchors. This is worth stating plainly — the real run's test partition is the
quietest fifth of the era, which is a further reason not to read its score.

### The item-8 pattern splits, and neither half is a result

Item 8 reported that all five shuffled controls beat real chronological order,
and read it as evidence against a temporal signal. Under the valid null:

* **Pooled, 5 of 5 controls still beat real order.** But `era_0`'s real pooled
  score is `0.392915`, well below chance, and any null sits near `0.5`. This
  says the real run is sub-chance pooled — the item-93 cross-era artifact — not
  that shuffling helps.
* **Stratified, 1 of 5 beats real order.** The real run is near the top of the
  null band rather than below it, which is the opposite of the item-8 reading.

Both halves rest on 1 held-out transition against 3–6, so neither is
interpretable. The honest conclusion is not that item 8 was backwards; it is
that **the comparison item 8 drew cannot be made on this record at all**, and
the appearance of a clean directional result came from a control scored on a
hundred times the evidence.

## What the valid null shows on `era_3`

`era_3` is the era worth reading: 19 held-out transitions against `era_0`'s 1,
8 scoreable cells against 1, and a held-out window that is close to
representative of the era.

**The null is evidence-matched here.** Held-out transitions, five seeds:

| | Real | Shift control | Shuffle control |
| --- | --- | --- | --- |
| `era_0` | 1 | 3, 4, 4, 5, 6 | 94, 98, 102, 111, 119 |
| `era_3` | 19 | 10, 13, 17, 17, 22 | 315, 317, 322, 336, 342 |

The shift controls bracket the real run. That is what the comparison needed and
never had.

Stratified balanced accuracy, 600 epochs both sides:

| Ablation | Real | Control mean | Control range | Controls beating real |
| --- | --- | --- | --- | --- |
| `seismic_only` | `0.552836` | `0.461116` | `0.400764`–`0.552511` | **0 of 5** |
| `seismic_vlf` | `0.539445` | `0.483559` | `0.404473`–`0.585987` | 1 of 5 |
| `vlf_only` | `0.532717` | `0.532322` | `0.476887`–`0.624882` | 2 of 5 |
| `all_features` | `0.404832` | `0.454386` | `0.374233`–`0.500000` | 3 of 5 |

### The item-8 pattern does not survive a matched null

Under the shuffle, **5 of 5** controls beat real order for every ablation, in
both eras. Under the shift the count is 0, 1, 2, or 3 of 5 depending on the
ablation, with no systematic direction. The clean directional result that item 8
reported — and that item 93 re-derived within eras — was a property of the
control, not of the data. It should not be cited again.

### `seismic_only` clears its null by a hair, and that is not a result

`seismic_only` is the one ablation no control beats. The margin is
`0.552836` against a best control of `0.552511`: a gap of `0.0003`, on 19 label
transitions, with the control spread running `0.40`–`0.55`. Real order ties its
best control. Counting "0 of 5" and reading it as separation would be exactly
the error [Target Design](target-design.md) was written to stop.

### Adding modalities makes it worse, consistently

Reading down the ablation table, every feature family added to seismic history
lowers the stratified score: `seismic_only` `0.5528` → `seismic_vlf` `0.5394` →
`all_features` `0.4048`. `vlf_only` sits at `0.532717` against a control mean of
`0.532322` — level with its own null to four decimal places.

This is consistent with items 102 and 104, and it is still not evidence of
absence. At 19 transitions the interval on every one of these numbers spans the
whole table. What can be said is narrower and worth saying plainly: **with a
null that is finally matched on evidence, no modality separates from chance, and
nothing improves on seismic history alone.**

## What this settles

Settled:

* **A usable null exists.** Sequence length and positive count preserved
  exactly per cell, run structure preserved apart from the wrap seam, held-out
  transitions bracketing the real run instead of exceeding it by two orders of
  magnitude.
* **The timestamp-shuffle results in items 8 and 93 are retired.** Not
  reversed — retired. The comparison they drew cannot be made with that control.
* **No ablation separates from its null on `era_3`.** The closest is
  `seismic_only`, by `0.0003`.

Not settled:

* Whether any modality carries signal. Nineteen held-out transitions cannot
  answer that, and this document does not claim to. The blocker recorded in
  item 104(d) is unchanged: calendar time.
* Whether the wrap seam matters at longer records. It adds at most a couple of
  transitions here; on a record with many more runs it would be worth replacing
  the single circular shift with a block bootstrap.

Nothing here supports any claim about earthquake prediction.
