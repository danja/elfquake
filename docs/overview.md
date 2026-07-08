# ELFQuake

ELFQuake is a research project for testing whether machine learning can identify useful earthquake-related signals from seismic, VLF radio, and astronomical data for Italy.

The central hypothesis is that VLF radio data, combined with seismic and astronomical context, may improve predictive signal enough to justify multimodal ML experiments.

The project should proceed cautiously: no prediction capability is assumed until it is demonstrated against reproducible baselines and held-out data.

## Documentation

Start with [Documentation Index](README.md), [Next Actions](next-actions.md), and [Analysis Report](report.md).

Keep source, simulation, modeling, and operations notes separate. Merge or archive stale smoke-run notes instead of adding more one-off documents.

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
