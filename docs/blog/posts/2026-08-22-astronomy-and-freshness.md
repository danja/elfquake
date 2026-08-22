---
date: 2026-08-22
categories:
  - Status
  - Data
  - Modeling
  - Italy
---

# The astronomical channel is real now, and it does not help

Two weeks ago the astronomy side of ELFQuake was a directory nobody was writing to. An audit on 2026-08-14 found eleven files across two days, against 699 Cumiana VLF captures across twenty-four; exactly two astronomical channels reached the Transformer, one of them a monthly constant repeated on all 11,020 rows and the other a count of files the collector happened to have written. The missing-modality mask that was supposed to flag all this was set from a value that was never blank, so it read "astronomy observed" on every row including the ten thousand with no astronomy at all.

That is all fixed. This post describes what the channel is now, and reports the first honest measurement of whether it does anything — which is the more important half, and the answer is no.

## What is actually collected

Three archives, on a daily timer of their own (`elfquake-space-weather.timer`) rather than the thirty-minute VLF cadence. The Kp/ap and F10.7 files are whole-history archives of roughly 16 MB and 2 MB and both publish at most once a day; refetching those every half hour would be wasteful locally and abusive upstream.

* **GFZ `Kp_ap_since_1932`** — planetary geomagnetic activity, three-hourly, CC BY 4.0.
* **Kyoto WDC Dst** — hourly ring-current index. Recent months are served only by the `realtime` tier, whose values get revised, so the tier is carried through normalization and the previous month is refetched on every run.
* **Spaceweather Canada F10.7** — solar radio flux, three Penticton observations a day at 17, 20 and 23 UT.

Alongside these sit computed ephemeris channels — lunar phase, distance, and a degree-two lunar-plus-solar tidal potential evaluated at the centre of the Italy bounding box. These are deterministic functions of UTC, not observations, and it matters below that they are.

Over the 817 prospective anchors currently in the table, every channel varies: `astro_kp` `0`–`7`, `astro_ap` `0`–`132`, `astro_dst_nt` `-146` to `+55` nT, `astro_f107` `94.8`–`258.1`. The window covers a real geomagnetic storm, so the geomagnetic channels have something to be tested against rather than a flat line.

## The alignment rule

The anchors are VLF capture times on a thirty-minute grid; the sources publish at three hours, one hour and one day. Bridging that gap is where the previous implementation went wrong, so the rule is now explicit.

Each channel takes the most recent reading whose observation interval has **closed** at or before the anchor — a zero-order hold, never interpolation. Interpolating between the readings either side of an anchor mixes a future reading into a feature that is supposed to be causal. A Kp bin covering 09:00–12:00 is not usable at 09:57.

Staleness is then a field rather than a footnote: every observed channel carries an `*_age_hours` companion. Past six hours for Kp and Dst, or seventy-two for F10.7, the hold is abandoned, the value goes blank, and `quality_missing_*` is set. Holding a three-day-old Kp reading forward would manufacture exactly the constant this work removes.

The aggregate flag `quality_missing_astro` fires only when Kp, Dst *and* F10.7 are all missing. Ephemeris channels are deliberately excluded from that test — they are always computable, so letting them satisfy it would pin the flag to zero forever, which is precisely what the old implementation did.

Full detail, including the ephemeris accuracy checks against Meeus, is in [Astronomy Alignment](../../astronomy-alignment.md).

## The fix had not actually reached the model

The alignment work landed on 2026-08-17 and was treated as done. It was not. The Transformer does not read the feature table; it reads sequence tensors materialized from it in a separate, manual step with no dependency check. Nothing rebuilt those. For thirteen days the fixture held nineteen aligned channels while the astronomy sequence the model actually opened still carried `astro_capture_count` and `astro_noaa_solar_cycle_f107_value` — the two dead channels the audit had condemned. Every Transformer run in that window trained on the dead pair.

After rematerializing, the astronomy sequence reports nineteen channels over 15,960 rows and 840 time steps, and — the part that matters — its masks now fire: the five Kp/ap channels read `present=0` on 152 rows at the end of the record, where the GFZ file has not yet closed its three-hour bin. Dst, F10.7 and the ephemeris channels are genuinely present throughout, so their masks staying clear is correct rather than broken.

The lesson is narrow and worth stating plainly: when checking whether a fix reached a model, look at the artifact the model opens, not the artifact the fix wrote.

## What it does

Two evaluations, both held out, both negative.

**Tabular, fixed-cell.** On the current design — one-day horizon, M≥2.0, cell-stratified, thresholds calibrated on training rows only, the late capture era — seismic history alone scores `0.552836`. Adding the twenty-one astronomy features scores exactly `0.500000`, and the exactness is the tell: the larger model predicts the negative class on all 1,976 test rows, having reached `0.613508` balanced accuracy on its training rows. Twenty-one features against nineteen label changes in the held-out period is the whole story. `full_multimodal` scores `0.431988` and the all-feature run `0.404832`. Every family added to seismic history lowers the score.

**Transformer, cross-region smoke.** The chained run — synthetic masked pretraining, Japan self-supervised continuation, chronological Italy fine-tuning on seismic + VLF + astronomy — completed end to end on 11,004 training and 2,752 test rows. Held-out balanced accuracy with all three modalities is `0.513960`. Masking astronomy at inference gives `0.511239`, masking seismic `0.511085`, masking VLF `0.493117`. A linear probe on the frozen representation scores `0.498439`.

Those differences are two to twenty thousandths on one seed, and the masking test measures how much a model trained on everything leans on a channel, not whether the modality adds value — that would need retraining without it. Read the whole block as "indistinguishable from `0.5`", because that is what it is.

## A caution for any future positive result

This one is worth more than the negatives above, because it is the trap a future run would fall into.

Over the 799 anchors of the labeled spatial table, five astronomical channels correlate strongly with the anchor index: `astro_f107` at `-0.880`, `astro_moon_phase_cos` at `+0.838`, `astro_moon_illuminated_fraction` at `-0.838`, `astro_moon_distance_km` at `-0.730`, `astro_tidal_potential_min` at `-0.604`. The geomagnetic channels sit below `0.26`.

No future information enters any row, so this is not a leak. But validation here is time-based — training rows early, test rows late — and a channel that is near-monotone over the record approximates an indicator for *which side of the split a row is on*. A model can use that to recover the era's base rate rather than any physical state, and a lift built that way would not survive a different split or a longer record.

The lunar channels are periodic and only look monotone because fifty-two days is under two lunations; they will decorrelate as the record grows. `astro_f107` will not — solar flux trends over months, and it will keep behaving like a date stamp for as long as the record is short compared with the solar cycle. Any astronomy result from here needs a date-proxy check and a matched null before it is treated as physical.

## Elsewhere, briefly

**The live collector was labeling real earthquakes as non-events.** The thirty-minute service updated VLF features against whatever event catalog happened to be on disk and never fetched INGV itself; only a manual run did. Between refreshes the catalog fell behind while VLF anchors kept accumulating, and every anchor in that gap was declared mature and labeled negative because no event could be found in a catalog that had stopped. On 2026-08-21 that put twenty-two real events' worth of false negatives into a held-out partition, and produced what had been the largest apparent VLF-era result in the project. After the fix, the late era's held-out balanced accuracy for seismic history fell from `0.652961` to `0.552836`. The service now fetches first, records what the catalog was successfully asked for, and tolerates a failed fetch rather than stopping VLF capture.

**The Cumiana colour-scale change is not removable.** Each capture embeds the colourbar it was drawn with, so decoding pixels back through it should make the two capture eras comparable. The features were built and every one of the 824 captures decodes. It does not help: hour-matched and day-median-reduced, a `-17.9 dB` step survives inversion, uniform from 200 Hz to 15 kHz at nearly three times the within-era day-to-day spread. Decoding recovers decibels as the receiver reported them, not decibels at the antenna, and a gain change alters the quantity plotted rather than the plot. The eras stay separate. See [Cumiana Colour-Scale Change](../../vlf-palette-shift.md).

**One earlier result has been retired outright.** The timestamp-shuffle control that underpinned the "shuffling beats real time order" finding was never a null: it carried between 94 and 342 held-out label changes against the real runs' 1 and 19, so it was scored on up to a hundred times the evidence. A circular shift of the whole labeled matrix preserves each cell's sequence length, positive count and run structure while destroying feature-label alignment. Under that control the pattern disappears — zero to three of five controls beat real order depending on the ablation, with no systematic direction. Details in [Within-Cell Null Control](../../shift-control.md).

**The staleness audit found six more instances.** A shared freshness guard now compares derived inputs against the event catalog and warns before any work is done. Two of the six were live rather than latent: the Transformer sequences described above, and both weekly forecast scripts, which defaulted their as-of date to the literal `2026-07-08` — a fresh catalog answering a frozen question, invisible to any timestamp check. See [Input Freshness](../../input-freshness.md).

## Where this leaves things

Astronomy has moved from *not measured* to *measured and null*. That is progress of a specific and limited kind: the earlier absence of a result was an artifact of a constant channel and a broken mask, and it now is not. The features are correct, varying, aligned, causally held, and honestly flagged when absent.

They also do not improve prediction at a one-day or seven-day horizon in the fixed-cell design, and at nineteen held-out label changes the design cannot currently distinguish a modality that helps from one that does not. That is the binding constraint on every modality here, not a fact about the Sun and the Moon. The queue says to re-run the ablation when a single capture era reaches held-out label changes in the low hundreds, and not before.

Nothing here is evidence for or against a relationship between astronomical or geomagnetic conditions and earthquakes. See [status](../../status.md), the [full report](../../report.md), and [next actions](../../next-actions.md).
