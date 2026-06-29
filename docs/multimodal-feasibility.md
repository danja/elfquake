# Multimodal Feasibility

The core project hypothesis is that VLF radio data may add useful predictive signal when combined with seismic and astronomical context.

## Early Questions

* Which VLF datasets are available for Italy-relevant stations or reception paths?
* Do VLF records have timestamps precise enough for seismic window alignment?
* What licensing permits reuse and derived features?
* Which astronomical and geomagnetic indexes are available at useful cadence?
* Can all sources be aligned to UTC windows without leaking future information?

## First Multimodal Window

Use the seismic smoke window as the first alignment target:

* region: Italy, with Central Italy subset
* window: 7 days
* seismic target: event with magnitude `>= 3.0`
* VLF features: Cumiana live image availability first, then image-derived signal summaries
* astronomical features: USNO lunar phase, NOAA solar and geomagnetic indexes

## Evaluation Rule

Build seismic-only baselines first, then compare:

1. seismic-only
2. seismic plus VLF
3. seismic plus astronomical
4. seismic plus VLF plus astronomical

Only treat multimodal features as useful if they improve held-out performance and calibration after ablation testing.
