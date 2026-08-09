---
date: 2026-08-09
categories:
  - Data
  - Italy
  - VLF
---

# A capture gap, not a precursor, before the Pisa earthquake

A M4.3 earthquake struck near Pisa on 4 August 2026 at 08:15 UTC, about 7 km from the city and 275 km from the Cumiana VLF receiver. Checking the two weeks before it against the Cumiana anomaly record looked briefly promising: the two highest-scoring days in the entire record fall inside that window.

They are an artifact. The collector recorded nothing between 16 and 29 July, and the anomaly scorer uses a 24-frame lookback, so every window straddling that 13-day gap reconstructs across a discontinuity and scores near the maximum. Those 21 windows average `0.9113`; the next 24 clean windows drop as a step function to `0.4074`. The high scores mark the collector restarting, not the ionosphere doing anything.

What the clean data shows runs the other way. The observed pre-event stretch averages `0.4127` against a `0.5429` pre-gap baseline, so the period before the earthquake was quieter than usual, and the strongest sustained activity in the record is *after* the event rather than before it.

The coverage problem is on its own disqualifying: with no captures from 17 to 28 July, the first eight of the fourteen pre-event days have no data, and more than half the window cannot be checked either way.

The useful outcome is a defect rather than a result. Every collector restart currently injects a false top-ranked anomaly into the record, which is precisely the failure mode that would manufacture a precursor claim from an outage. A gap-aware guard that flags windows spanning a capture gap longer than the lookback is now queued before this scorer is used for event association again.

This remains a single-event retrospective with one station, no directional information, and no matched control windows. It is not evidence for or against a VLF-earthquake relationship.

See the [full analysis](../../pisa-2026-08-04-vlf-check.md) and [next actions](../../next-actions.md).
