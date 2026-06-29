# System Architecture

The architecture should stay modular so data sources, features, and models can be replaced independently.

## Pipeline

1. Ingest raw source data.
2. Validate schemas, timestamps, and required fields.
3. Normalize Italy-scoped records into consistent units and UTC time.
4. Extract features over fixed temporal windows and Italian spatial regions.
5. Run exploratory analysis and quality checks.
6. Train baseline and candidate models.
7. Evaluate with backtests and holdouts.
8. Publish reports, artifacts, and limitations.

## Boundaries

* Keep raw data immutable.
* Keep source connectors separate from normalization logic.
* Keep feature generation separate from model training.
* Keep evaluation independent of training code.
* Keep the Italy scope explicit in dataset metadata and reports.

## Outputs

The minimum useful outputs are normalized datasets, feature tables, model artifacts, evaluation reports, and run metadata.
