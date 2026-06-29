# Modeling Strategy

Modeling should begin only after data coverage and target definitions are clear.

## Principles

* Start with naive baselines, then simple statistical or classical ML models.
* Define prediction targets before selecting algorithms.
* Use explicit prediction windows and Italian geographic regions.
* Compare seismic-only models with seismic plus VLF and astronomical features.
* Report uncertainty and calibration, not only point predictions.
* Treat negative results as valid research outputs.

## Candidate Targets

* Event occurrence within an Italy time and location window.
* Magnitude threshold exceedance.
* Relative risk score compared with historical baseline rates.

## Constraints

Avoid claims of earthquake prediction unless the system beats strong baselines on held-out data and the error profile is scientifically defensible.

Multimodal models must be evaluated with ablations so any gain from VLF or astronomical features is measured directly.

Use [Multimodal Window Schema](multimodal-window-schema.md) as the table contract for baseline and ML inputs.
