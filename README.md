# ELFQuake

ELFQuake is an research project testing whether VLF radio data, combined with seismic and astronomical context, can add useful signal to earthquake-related predictive modeling.

Status: acquisition and feature pipelines are running; only smoke-model trials have been performed; no prediction capability is claimed.

Current work:

* INGV seismic event acquisition and normalization for Italy/Central Italy.
* Cumiana VLF live image capture through systemd.
* Pixel-derived VLF image features from spectrogram JPEGs.
* Astronomical and space-weather archive normalization.
* Prospective VLF-anchored rows with pending target labels.
* Dependency-free model readiness, logistic smoke, and ablation smoke reports.

Important docs:

* [Overview](docs/overview.md)
* [Next Actions](docs/next-actions.md)
* [VLF Coverage](docs/vlf-coverage.md)
* [Prospective Labeling](docs/prospective-labeling.md)
* [Initial Model Trials](docs/initial-model-trials.md)
* [Systemd Service](docs/systemd-service.md)

The first prospective VLF rows become labelable after `2026-07-06T09:57:24Z`.
