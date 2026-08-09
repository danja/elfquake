# Pisa M4.3 VLF Check (2026-08-04)

Retrospective check of the Cumiana VLF record for anomalies in the two weeks
before the 2026-08-04 M4.3 earthquake near Pisa.

**Result: no supportable precursor signal.** The only strongly elevated
anomaly scores in the pre-event window are a capture-gap artifact, and the
genuinely observed pre-event period is quieter than the record baseline.

This is a descriptive single-event check with no controls, no matched
non-event windows, and no correction for multiple comparisons. It is not
evidence for or against a VLF-earthquake relationship.

## Target event

| Field | Value |
| --- | --- |
| INGV event id | `46769822` |
| Origin time | `2026-08-04T08:15:18.640Z` |
| Location | 43.6925N, 10.3263E (~7 km from Pisa) |
| Depth | 8.2 km |
| Magnitude | 4.3 ML |
| Distance from Cumiana receiver | ~275 km |

A M2.4 aftershock at 43.7057N, 10.3598E followed on `2026-08-09T06:37:27Z`.

The receiver at Cumiana (44.98N, 7.38E) is a single station 275 km from the
epicentre. It records a JPG spectrogram, not a calibrated field measurement,
and has no directional information. Nothing in this setup can localize a
source to Pisa even if an anomaly were present.

## Coverage blocker

The two-week pre-event window runs `2026-07-21T08:15Z` to `2026-08-04T08:15Z`.
Capture counts per day:

| Date | Captures |
| --- | --- |
| 2026-07-16 | 1 |
| 2026-07-17 .. 2026-07-28 | **0** |
| 2026-07-29 | 19 |
| 2026-07-30 | 29 |
| 2026-07-31 | 35 |
| 2026-08-01 | 48 |
| 2026-08-02 | 40 |
| 2026-08-03 | 24 |
| 2026-08-04 | 37 |

**The first 8 of the 14 pre-event days have no data at all.** Only
2026-07-29 onward is observed, and 2026-08-06 is also missing. Any claim about
the two-week window is therefore unfalsifiable over more than half its span.

## The gap artifact

Scored with `./scripts/score-real-vlf-anomaly-forecast.sh` (558 windows,
`lookback-steps=24`, `stride=1`, self-supervised reconstruction + embedding
novelty).

The two highest-scoring days in the entire record fall inside the pre-event
window, which looks suggestive until the window structure is checked:

| Date | n | mean | max | max-rank |
| --- | --- | --- | --- | --- |
| 2026-07-29 | 19 | 0.9045 | **0.9877** | 1/19 |
| 2026-07-30 | 29 | 0.4463 | **0.9765** | 2/19 |
| 2026-07-31 | 35 | 0.5303 | 0.8038 | 12/19 |
| 2026-08-01 | 48 | 0.3600 | 0.7601 | 15/19 |
| 2026-08-02 | 40 | 0.3769 | 0.6132 | 18/19 |
| 2026-08-03 | 24 | 0.2940 | 0.7982 | 14/19 |
| 2026-08-04 | 37 | 0.8427 | 0.9114 | 7/19 |

With a 24-frame lookback, every window ending within 24 frames of the
collector restart still contains pre-gap frames from 13 days earlier. Those
windows are reconstructing across a discontinuity, which mechanically produces
near-maximal novelty. The scores collapse as a step function the moment the
lookback buffer flushes:

```
2026-07-29T17:21Z   0.9877   <- gap-spanning
2026-07-30T05:53Z   0.9742   <- gap-spanning
2026-07-30T06:23Z   0.9765   <- last gap-spanning window
2026-07-30T06:53Z   0.3823   <- first clean window
2026-07-30T07:23Z   0.2691
```

* 21 gap-contaminated windows: mean `0.9113`
* next 24 clean windows: mean `0.4074`

This is an instrumentation artifact of the collector restart, not a
geophysical observation.

## What the clean data shows

| Period | n | mean | p90 | max |
| --- | --- | --- | --- | --- |
| Pre-gap baseline (Jun 30 - Jul 16) | 256 | 0.5429 | 0.8285 | 0.9395 |
| Clean pre-event (Jul 30 07:00 - Aug 4 08:15) | 179 | **0.4127** | 0.7063 | 0.8991 |
| Post-event (Aug 4 08:15 - Aug 8) | 101 | **0.7214** | 0.9013 | 0.9114 |

The observed pre-event period is **quieter** than the baseline, and the
highest sustained activity in the record is *after* the earthquake. Both
observations point away from a precursor reading. The post-event elevation is
itself more likely to reflect ordinary ionospheric or local-noise variation
than aftershock coupling; it has not been tested against controls.

## Follow-ups

1. Add a gap-aware guard to `score-sequence-anomalies`: flag or invalidate any
   window whose lookback span crosses a capture gap longer than the lookback
   interval. Until this exists, every collector restart injects a false
   top-ranked anomaly into the record.
2. Fix collector continuity before attempting further event association. The
   13-day July gap and the missing 2026-08-06 make single-event retrospectives
   uninterpretable. Track with `./scripts/report-vlf-capture-gaps.sh`.
3. Do not run further single-event retrospectives without matched non-event
   control windows drawn from the same coverage regime. Picking the largest
   score in a window that contains a known artifact is not a test.
