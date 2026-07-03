# ELFQuake

ELFQuake is a research project for testing whether machine learning can identify useful earthquake-related signals from seismic, VLF radio, and astronomical data for Italy.

The central hypothesis is that VLF radio data, combined with seismic and astronomical context, may improve predictive signal enough to justify multimodal ML experiments.

The project should proceed cautiously: no prediction capability is assumed until it is demonstrated against reproducible baselines and held-out data.

## Documentation

* [Roadmap](roadmap.md)
* [Development Environment](development-environment.md)
* [Data Sources](data-sources.md)
* [Data Acquisition Code](data-acquisition-code.md)
* [Acquisition Smoke Run](acquisition-smoke-run.md)
* [Batch Run 2026-06-29](batch-run-2026-06-29.md)
* [Source Inventory](source-inventory.md)
* [Multimodal Feasibility](multimodal-feasibility.md)
* [Multimodal Window Schema](multimodal-window-schema.md)
* [Multimodal Smoke Row](multimodal-smoke-row.md)
* [VLF Feasibility](vlf-feasibility.md)
* [Cumiana Live VLF](vlf-cumiana-live.md)
* [VLF Capture Manifest](vlf-capture-manifest.md)
* [Astronomical Feasibility](astronomical-feasibility.md)
* [NOAA Archive Feasibility](noaa-archive-feasibility.md)
* [Backfill Planning](backfill-planning.md)
* [Feature Extraction](feature-extraction.md)
* [Signal Shape Comparison](signal-shape-comparison.md)
* [Target Labeling](target-labeling.md)
* [Training Windows](training-windows.md)
* [Design Matrix](design-matrix.md)
* [Archive Normalization](archive-normalization.md)
* [Connector Notes](connector-notes.md)
* [Event Schema](event-schema.md)
* [Normalization](normalization.md)
* [Sample Dataset](sample-dataset.md)
* [Smoke Dataset](smoke-dataset.md)
* [Derived Datasets](derived-datasets.md)
* [Exploratory Report](exploratory-report.md)
* [Smoke Exploratory Report](exploratory-report-smoke.md)
* [Baseline Targets](baseline-targets.md)
* [Baseline Input](baseline-input.md)
* [Smoke Baseline Run](baseline-run-smoke.md)
* [System Architecture](system-architecture.md)
* [Systemd Service](systemd-service.md)
* [Pre-Real-Data Checklist](pre-real-data-checklist.md)
* [Modeling Strategy](modeling-strategy.md)
* [Model Candidates](model-candidates.md)
* [Model Readiness](model-readiness.md)
* [Initial Model Trials](initial-model-trials.md)
* [Evaluation](evaluation.md)
* [Next Actions](next-actions.md)

## Development Phases

1. Data Connection
2. Data Analysis
3. Simulation Construction
4. Model Construction
5. Model Evaluation
6. End-to-end System Construction
7. System Evaluation
8. Deployment

## Components

The first components should be small and replaceable:

* data connectors
* normalized datasets
* exploratory notebooks or scripts
* feature extraction jobs
* baseline models
* evaluation reports
