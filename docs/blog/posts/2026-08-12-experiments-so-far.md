---
date: 2026-08-12
categories:
  - Status
  - Modeling
  - Italy
  - Japan
---

# Where the experiments stand

ELFQuake has been running as a live pipeline for about six weeks. This is a summary of what has been tried, what came out of it, and what the system currently is. The short version: the infrastructure works and is reproducible, several of the results that looked encouraging turned out to be artifacts of the pipeline rather than of the ionosphere, and no model has yet beaten its own control.

## What is being collected

Four streams feed the Italy track. INGV earthquake events are pulled from the public FDSN service and normalized to UTC with source identifiers and uncertainty fields preserved; the current catalog holds 5,013 events through 2026-08-08. Cumiana VLF spectrograms are captured live as JPEGs on a 30-minute cadence by a systemd timer, with image and metadata features derived from them. NOAA SWPC space-weather and USNO moon-phase JSON provide the astronomical channel. A Numba CPU sandpile simulation generates synthetic avalanche and piezo-like signals for controlled experiments.

There is a second, research-only Japan track using ISEE Moshiri CDF files and USGS seismic history, kept separate because its use terms are narrower.

## The experiments

**Seismic-only and transfer baselines.** The main real-data checkpoint is a fixed-cell target: does a 1.5-degree Italy cell contain an M2.5+ event in the following week? A small CPU model pretrained on synthetic avalanche catalogs, then fine-tuned chronologically on real INGV history, scores held-out balanced accuracy `0.667730` and precision `0.279167`. Its control — a historical spatial-rate lookup fitted on the training period alone — scores `0.687589` and `0.322751`. The model is below its own control on both metrics. VLF and astronomy enter this run as missing-modality masks, so it is a seismic-history baseline and nothing more.

**The multimodal spatial baseline.** A separate table expands each VLF capture time across 19 Italy cells; it currently has 11,020 rows with 7,258 labeled across 580 anchor times. A grouped-time logistic ablation on it scored `0.412604` calibrated balanced accuracy — below the 0.5 always-negative baseline. Timestamp-permutation controls, which destroy temporal ordering while preserving each timestamp's spatial pattern, scored *higher* than the real chronological ordering. That is the opposite of a predictive signal and prompted a diagnostic rather than a model change.

**Synthetic simulation and precursor search.** Considerable effort went into making sandpile output resemble real catalogs and into finding a precursor inside the simulation itself. Catalog calibration works reasonably: a five-episode central-Italy corpus reaches a rate ratio of `0.924` and magnitude Wasserstein distance `0.046`. The precursor search has not. A delayed-failure damage mechanism produced a genuine causal lead across nine episodes (AUC `0.652315`), but a matched Transformer ablation on the same trajectories scored `0.599648` *without* the damage channels and `0.586848` with them. Two-stage maturation, per-receiver damage readouts, decay and reset sweeps, and a bounded stress reservoir all failed their causal confirmation gates. The stress reservoir's release-aware diagnostic found only `5.9%` of releases followed by positive local excess activity, with a median lag of 96 steps — weak and spatially broad.

**Self-supervised VLF.** With real labels one-class or sparse, label-free representation learning is the default path. A sequence model reconstructs Cumiana image features and scores each window for novelty. It runs, but reconstruction quality is not earthquake relevance, and the anomaly record needed a correction described below.

**Event retrospectives.** Two were run against actual earthquakes. For the 2026-07-28 Kyushu event, Moshiri VLF scores on the event day ranged `1.529`–`5.418` with the maximum *after* the mainshock, while 27 July reached `6.305` beforehand — elevated, but not event-specific. For the 2026-08-04 M4.3 near Pisa, the answer was clearer and negative; see the [earlier post](2026-08-09-pisa-vlf-check.md).

## What the experiments found

The most useful results have been defects, not signals.

Three separate negative results from 2026-08-09 turned out to share one cause. The Cumiana record contains two dense capture eras — 2026-06-28 to 2026-07-05, and 2026-07-28 onward — separated by a collector outage with gaps of 104, 127 and 308 hours. The chronological train/test boundary falls exactly on that outage, so the model trains on one era and tests on the other. Restricted to a single era, the sub-chance score disappears: `0.655320` in the early era and `0.575000` in the late one. Shuffling helped because it removed a shift the real split imposes. The shift lives in image content, not capture cadence — the largest standardized differences are all intensity, band and colour-ratio features, consistent with a gain or colour-scale change across the outage. Full detail is in [Capture-Era Shift](../../capture-era-shift.md).

Cleaning up the confound did not rescue the model. Within-era permutation controls at matched epochs still beat real time order in three of five runs per era. And the decisive control is blunt: dropping the two cell-coordinate columns collapses both eras to exactly `0.500000`. The fixed-cell model is a static per-cell base-rate lookup with no temporal component at all.

The anomaly scorer had the same class of problem. Its 24-frame lookback reconstructed straight across capture gaps, so every window straddling an outage scored near maximum. A gap-aware guard now flags these: 66 of 558 windows span a gap, and **43 of the 101 historical `>=0.8` alerts were gap artifacts**. Gap-spanning windows average `0.7908` against `0.5143` for clean ones. Any earlier alert count from this scorer is inflated.

Two staleness defects were also found. A refresh script rebuilt only its date-scoped catalogs and never the `all_available` one that six downstream analyses read, freezing it at 2026-07-07 for a month; the transfer trial reported an identical `0.693435` across three "refreshed" runs, which had been read as stability. A second script rebuilt its inputs only when missing. Both are fixed, and the working rule now is that a score reproducing to six decimal places across a refresh is a staleness signal, not a stability result. A latent standardization bug — a near-constant column with variance `2e-28` treated as varying — was fixed alongside them.

## Current status

The pipeline is real and reproducible: acquisition, normalization, feature extraction, alignment, training, ablation, and a demonstration seven-day event list and map all run end to end on CPU, with regression tests over the defect classes found so far. That is the deliverable at this stage.

The science is at feasibility, not evidence. No multimodal model has beaten a seismic-only or historical-rate control on held-out real data. The blocking constraint is coverage: the late VLF era has four days of labeled window and 399 test rows, astronomy is entirely absent from it, and Japan has 9 VLF-observed windows out of 133. Strong earthquakes are rare enough that a credible test needs far more overlapping record than exists.

Next in order: determine from raw spectrograms whether the low-band change across the outage is instrumental or physical, since pooling the two eras is unsafe if it is instrumental; reconsider whether the fixed-cell target can express anything beyond per-cell base rate; keep extending capture coverage; and audit the remaining scripts for the staleness pattern.

Nothing here is evidence for or against a VLF-earthquake relationship. See [status](../../status.md), the [full report](../../report.md), and [next actions](../../next-actions.md).
