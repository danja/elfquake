# Next Actions

What to do next, and why. The record of what has already been tried and what it
found is in [Findings and Decisions Log](findings-log.md), whose numbered items
are cited from here and from the rest of the documentation.

Nothing in this project has demonstrated earthquake prediction. Every entry
below is either a measurement to make or a constraint that stops one from being
made.

## The blocking constraint

**Held-out label variation, not features, model capacity, or CPU time.**

The fixed-cell target uses 30-minute anchors against a multi-day horizon, so
consecutive target windows for one cell overlap by more than 99% and the rows
inside a cell are near-copies. The honest sample size is the number of times a
cell's label *changes* between consecutive anchors, and every report now prints
that count before any score
([log item 103](findings-log.md), [Target Design](target-design.md)).

Current counts, held out:

| Partition | Transitions | Scoreable cells |
| --- | --- | --- |
| `era_0` (2026-06-28 to 2026-07-06) | 1 | 1 of 19 |
| `era_3` (2026-07-28 onward) | 19 | 8 of 19 |
| pooled, 7-day horizon | 5 | 5 of 19 |

At those counts a swing of ten points is noise, and the two capture eras cannot
be pooled because a `17.9 dB` receiver-level step separates them
([log item 96](findings-log.md), [Cumiana Colour-Scale Change](vlf-palette-shift.md)).
No modality can be tested until this number reaches the low hundreds inside a
single era, which is a matter of calendar time.

## Now

1. **Keep collecting.** `elfquake-prospective.timer` every 30 minutes for VLF
   and INGV, `elfquake-space-weather.timer` daily for the geomagnetic and solar
   archives, `elfquake-japan-vlf.timer` for the research-only Japan track. This
   is the only action that moves the blocking constraint.
2. **Ask the Cumiana operator whether receiver gain changed between 2026-07-06
   and 2026-07-11.** Images alone cannot separate a front-end gain reduction
   from a quieter ionosphere, and this is the only route to settling it
   ([log item 105(c)](findings-log.md)).
3. **Re-run `./scripts/diagnose-vlf-palette-shift.sh` after each capture
   refresh.** A third palette variant would split the record again.
4. **Re-run the fixed-cell evaluation through the
   `elfquake-fixed-cell-evaluation` skill when one era reaches held-out
   transitions in the low hundreds** — and not before. The skill pins the
   as-of and catalog-end stamps so a run is reproducible.

## After each data refresh

Run these, and read the caveat attached to each.

| Command | Caveat |
| --- | --- |
| `./scripts/run-real-transfer-trial.sh` | Seismic-history baseline; VLF and astronomy enter as missing masks. Currently `0.671167` against a `0.69222` historical-rate control — below its own control. |
| `./scripts/evaluate-italy-spatial-baseline.sh` | Set `STRATIFY_FIELD=target_cell_id`. Pools both capture eras, so its VLF features are not on one scale; cite the per-era numbers instead. |
| `./scripts/report-italy-data-coverage.sh` | Warns if the anomaly scores predate the catalog. |
| `./scripts/run-transfer-experiments.sh` | Compares historical rate, real-only init, synthetic transfer, and rolling-origin folds. |

A script that reads a derived model artifact now checks its age against the
event catalog first ([Input Freshness](input-freshness.md)). **A score that
reproduces to six decimal places across a refresh is a staleness signal, not a
stability result.**

## Rules for reading any result

* **Print the transition count before the score.** A balanced accuracy without
  it is not interpretable.
* **Use the circular-shift null, not a timestamp shuffle.** The shuffle carries
  up to a hundred times the evidence of the run it is meant to control and is
  not a null ([Within-Cell Null Control](shift-control.md)).
* **Check the coordinate control.** Removing the two cell-coordinate columns has
  so far collapsed the fixed-cell model to exactly `0.5`, which means the score
  was per-cell base rate.
* **Check astronomy channels for date proxying.** Five of them correlate above
  `0.6` with the anchor index over the current record, so under a time-based
  split they approximate an indicator for which side of the split a row is on
  ([Astronomy Alignment](astronomy-alignment.md)).
* **Do not pool capture eras** for anything derived from Cumiana pixel colour.
* **Time-based validation only.** Training data must precede validation data.

## Held open, no action available

* **Per-cell rate residual and single-regional-target designs.** Two candidate
  replacements for the fixed-cell contract. Both need enough label variation to
  tell them apart, so neither can be tested yet
  ([log item 103](findings-log.md)).
* **Japan cross-region training.** 9 VLF-observed windows against 124 missing.
  Interface checks pass; there is nothing to train on. Research-use-only terms
  apply to all ISEE data.
* **Synthetic precursor search.** Delayed-failure damage, two-stage maturation,
  per-receiver readouts, and a bounded stress reservoir have each failed their
  causal confirmation gates. Do not tune scalar parameters further without a
  stronger mechanism ([log items 55-62](findings-log.md)).

## Where things are written up

| Concern | Document |
| --- | --- |
| The numbered record | [Findings and Decisions Log](findings-log.md) |
| Target contract and evidence counting | [Target Design](target-design.md) |
| The matched null control | [Within-Cell Null Control](shift-control.md) |
| Capture-era split and the dB step | [Cumiana Colour-Scale Change](vlf-palette-shift.md), [Capture-Era Shift](capture-era-shift.md) |
| Astronomy channels and alignment | [Astronomy Alignment](astronomy-alignment.md) |
| Stale-input defects and the guard | [Input Freshness](input-freshness.md) |
| Mistakes and their prevention | `MISTAKES.md` |
