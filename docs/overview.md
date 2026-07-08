# ELFQuake Overview

ELFQuake is a research project for testing whether machine learning can identify useful earthquake-related signals from seismic, VLF radio, and astronomical data for Italy.

The central hypothesis is that VLF radio data, combined with seismic and astronomical context, may improve predictive signal enough to justify multimodal ML experiments.

The project should proceed cautiously: no prediction capability is assumed until it is demonstrated against reproducible baselines and held-out data.

## Documentation Entry Points

Start with:
*   [Documentation Index](README.md) - Map of all documents
*   [Next Actions](next-actions.md) - Immediate work queue
*   [Analysis Report](report.md) - Current analysis status and caveats
*   [Processing Graph](processing-graph.md) - Current source-to-model data flow
*   [Command Steps](steps.md) - Runnable command reference

Keep source, simulation, modeling, and operations notes separate.

---

## Development Phases and Roadmap

The project moves from data feasibility to modeling phases. Each phase requires reproducible outputs:

1.  **Data Connection**: Confirm access to Italy-scoped seismic, VLF radio, and astronomical sources.
2.  **Data Analysis**: Inspect coverage, gaps, noise, bias, and timestamp alignment across all modalities.
3.  **Simulation Construction**: Create simple replay and backtesting workflows.
4.  **Model Construction**: Start with naive and classical baselines before complex ML.
5.  **Model Evaluation**: Compare against held-out time and geography splits.
6.  **End-to-end System**: Connect ingestion, feature generation, training, and reports.
7.  **System Evaluation**: Verify repeatability, drift, latency, and operational costs.
8.  **Deployment**: Publish only validated analysis outputs and documented limitations.

### Roadmap Milestone Order
*   Build a reproducible data inventory.
*   Create normalized Italy seismic, VLF, and astronomical datasets.
*   Produce single-source and cross-source exploratory reports.
*   Define baseline targets and metrics.
*   Run seismic-only baselines before adding multimodal models.

---

## System Architecture

The architecture is modular so data sources, features, and models can be replaced independently.

### Processing Pipeline
1.  **Ingest** raw source data.
2.  **Validate** schemas, timestamps, and required fields.
3.  **Normalize** Italy-scoped records into consistent units and UTC time.
4.  **Align** seismic events, VLF signals, and astronomical indexes into fixed temporal windows.
5.  **Extract** modality-specific and cross-modal features.
6.  **Analyze** exploratory reports and quality checks.
7.  **Train** seismic-only baselines before multimodal candidate models.
8.  **Evaluate** with backtests, holdouts, and ablations.
9.  **Publish** reports, artifacts, and limitations.

### Architectural Boundaries and Constraints
*   **Immutability**: Keep raw source data unchanged.
*   **Separation of Concerns**:
    *   Keep source connectors separate from normalization logic.
    *   Keep feature generation separate from model training.
    *   Keep evaluation independent of training code.
*   **Scope and Provenance**:
    *   Keep the Italy scope explicit in dataset metadata and reports (limit project data to Italy).
    *   Preserve modality provenance explicit in every feature table.

### Minimum Outputs
The minimum useful outputs are normalized datasets, feature tables, model artifacts, evaluation reports, and run metadata.

