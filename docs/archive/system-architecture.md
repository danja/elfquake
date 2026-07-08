# System Architecture

The architecture should stay modular so data sources, features, and models can be replaced independently.

## Pipeline

1. Ingest raw source data.
2. Validate schemas, timestamps, and required fields.
3. Normalize Italy-scoped records into consistent units and UTC time.
4. Align seismic events, VLF signals, and astronomical indexes into fixed temporal windows.
5. Extract modality-specific and cross-modal features.
6. Run exploratory analysis and quality checks.
7. Train seismic-only baselines before multimodal candidate models.
8. Evaluate with backtests, holdouts, and ablations.
9. Publish reports, artifacts, and limitations.

## Boundaries

* Keep raw data immutable.
* Keep source connectors separate from normalization logic.
* Keep feature generation separate from model training.
* Keep evaluation independent of training code.
* Keep the Italy scope explicit in dataset metadata and reports.
* Keep modality provenance explicit in every feature table.

## Outputs

The minimum useful outputs are normalized datasets, feature tables, model artifacts, evaluation reports, and run metadata.
