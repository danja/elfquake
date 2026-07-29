# Cross-Region Generative Smoke Test

## Purpose

Exercise the intended transfer path end to end:

1. Pretrain the shared Transformer representation on substantial synthetic multimodal sequences.
2. Continue representation training on Japan ISEE VLF and seismic sequences.
3. Fine-tune the event head on Italy data, holding out the latest mature Italy week.
4. Emit selected probabilities over fixed Italy cells and render them beside the actual held-out events.

This is a pipeline and visualization test, not a prediction experiment. It must not be presented as evidence of earthquake forecasting ability.

## Data Policy

The core target remains Italy-scoped. Japan ISEE data and derived features are used only for scientific research, with source provenance and the research-use restriction preserved. Japan is an intermediate representation-learning domain, not an additional Italy label source.

## Training Stages

* **Synthetic:** masked sequence reconstruction plus optional synthetic event-list supervision using the existing seismic, direct-avalanche, piezo/VLF, and summary branches.
* **Japan:** self-supervised continuation on native Moshiri CDF sequence features. Do not use Japan target labels while the observed VLF coverage remains sparse.
* **Italy:** supervised fine-tuning on fixed 1.5-degree Italy cells with seismic history, VLF image features, astronomy features, and explicit missing-data masks.

The encoder and modality adapters are retained between stages. The Italy event head is reinitialized or fine-tuned only after the Italy feature schema is checked against the checkpoint schema.

## Holdout

Use the latest mature Italy VLF-anchored week. Exclude its target horizon and all later rows from training and calibration. The current 2026-07-29 capture is pending and must not be used as a held-out label until its seven-day horizon matures. Select thresholds using earlier Italy rows only.

## Output

Write a stable event-list CSV containing forecast week, cell centre, probability, and rank. Render predicted cell centres as red crosses and actual INGV events above the selected magnitude threshold as blue circles, using the existing Italy basemap and magnitude scaling.

Report the historical-rate and always-negative controls, train/test class counts, selected threshold, and missing-modality rates beside the image. Do not report the map as a forecast.

## Gates

The smoke test is successful when all stages run without schema leakage, Japan data remains provenance-labelled, the Italy holdout is chronological, and the map contains both actual and model-derived layers. It fails as a model-evidence test if the holdout is one-class, if Japan labels leak into Italy targets, or if thresholds are selected on the held-out week.
