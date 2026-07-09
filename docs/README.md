# Documentation Index

Use this as the master directory map for the `docs/` folder. Prefer updating an existing document in the appropriate group before creating a new one.

---

## 1. Project Entry Points
*   **[Overview](overview.md)**: Project scope, development phases, and system architecture boundaries.
*   **[Next Actions](next-actions.md)**: The immediate project work queue, priorities, and roadmap log.
*   **[Analysis Report](report.md)**: Current analysis status, experimental outcomes, and caveat logs.
*   **[Command Steps](steps.md)**: Full runnable command reference for pipelines, features, and model workflows.
*   **[Development Environment](development-environment.md)**: Local Python virtual environment setup, systemd production timer install, and deployment checklist.
*   **[Processing Graph](processing-graph.md)**: Diagram of the source-to-model data flow for synthetic and real paths.

---

## 2. Data Sources and Normalization
*   **[Source Inventory and Onboarding](source-inventory.md)**: Directory of Italy-relevant data endpoints, licensing terms, and onboarding checklist.
*   **[VLF Radio Feasibility](vlf-feasibility.md)**: Findings, endpoints, capture manifests, storage rules, and image/audio extraction details for Cumiana and Abelian VLF data.
*   **[Astronomical and Geomagnetic Feasibility](astronomical-feasibility.md)**: Solar cycle, lunar phase, and Kp/ap planetary index archives and live feeds.
*   **[Event Schema](event-schema.md)**: Specifications and fields for normalized earthquake events.
*   **[Source Data Normalization](normalization.md)**: Rules, schemas, and commands to convert raw INGV and space-weather files into normalized tables.

---

## 3. Modeling and Evaluation
*   **[Modeling and Evaluation Strategy](modeling-strategy.md)**: Principles, dynamic target labeling, model readiness checks, validation splits, and metrics.
*   **[Baseline Targets](baseline-targets.md)**: Reference targets, metrics, and chronological split rules for the naive historical-rate baseline.
*   **[Forecast Interface](forecast-interface.md)**: Stable weekly event-list input/output contract for trial and future learned scorers.
*   **[Success Criteria](success-criteria.md)**: Staged pass/fail criteria for scaffold, synthetic utility, real readiness, and prediction claims.
*   **[Model Candidates](model-candidates.md)**: Tabular, GRU, and Transformer neural candidate architectures.
*   **[Model Interface Shape](model-interface-shape.md)**: Tensor layouts, mask configurations, and dataset constraints.
*   **[Model Comparison](model-comparison.md)**: Diagnostic comparisons of baselines against synthetic/real sequence model runs.
*   **[Model Scaling Requirements](model-scaling-requirements.md)**: CPU limits, neural parameters, and sequence scaling constraints.

---

## 4. Simulation
*   **[Sandpile Simulation](sandpile-simulation.md)**: CPU-only sandpile model design, terrain structures, sensors, and avalanche features.
*   **[Simulation Time Scale](simulation-time-scale.md)**: Time step to physical UTC mapping rules.
*   **[Signal Shape Comparison](signal-shape-comparison.md)**: Visual diagnostics comparing simulated piezo features against VLF shapes.

---

## 5. Operations
*   **[Prospective Labeling](prospective-labeling.md)**: Procedure and scheduling for label rollover checks and prospective table refreshes.

---

## 6. Archived Documents
Stale logs, early planning checklists, and redundant summaries have been relocated to the **[archive/](archive/)** directory:
*   `archive/acquisition-smoke-run.md`: First live captures test summary.
*   `archive/archive-normalization.md`: Specific normalizer notes.
*   `archive/backfill-planning.md`: Long-run pull planning.
*   `archive/baseline-input.md`: Stale baseline schema rules.
*   `archive/baseline-run-smoke.md`: First baseline trial output.
*   `archive/batch-run-2026-06-29.md`: Ingest log from June 2026.
*   `archive/connector-notes.md`: Tested endpoint queries and failures.
*   `archive/data-acquisition-code.md`: Early Python CLI command references.
*   `archive/data-sources.md`: Checklist of candidate datasets.
*   `archive/derived-datasets.md`: Inventory of derived files.
*   `archive/design-matrix.md`: Legacy table specifications.
*   `archive/exploratory-report-smoke.md`: First usability counts.
*   `archive/feature-extraction.md`: Early visual feature extraction stubs.
*   `archive/initial-model-trials.md`: Early ablation scores on tiny samples.
*   `archive/model-readiness.md`: Old class balance logs.
*   `archive/model-targets.md`: Archive target guidelines.
*   `archive/multimodal-smoke-row.md`: Single-row test schematic.
*   `archive/noaa-archive-feasibility.md`: Early NOAA archive assessments.
*   `archive/pre-real-data-checklist.md`: Old deployment prep checklist.
*   `archive/roadmap.md`: Early roadmap layout.
*   `archive/smoke-dataset.md`: First INGV text pull details.
*   `archive/system-architecture.md`: Pipeline boundaries.
*   `archive/systemd-service.md`: Linux service install guides.
*   `archive/target-labeling.md`: Dynamic labeling parameters.
*   `archive/training-windows.md`: Legacy 7-day window details.
*   `archive/vlf-abelian.md`: Incomplete Abelian Natural Radio details.
*   `archive/vlf-capture-manifest.md`: Cumiana polling manifests.
*   `archive/vlf-coverage.md`: Multi-step feature table details.
*   `archive/vlf-cumiana-live.md`: Live image endpoints reference.
