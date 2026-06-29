# AGENTS.md

Guidance for agents working in this repository.

## Project Scope

ELFQuake is an Italy-scoped research project for testing whether seismic, natural radio, and astronomical data contain useful earthquake-related signals.

Do not claim earthquake prediction capability unless it is demonstrated against reproducible baselines and held-out data.

## Current Priority

Prioritize feasibility and reproducibility:

1. Verify data source access end to end.
2. Build a small Central Italy sample dataset.
3. Produce an exploratory data report.
4. Run a naive historical-rate baseline.
5. Compare results against documented evaluation criteria.

## Documentation

Keep docs concise and modular. Prefer one document per concern under `docs/`.

Start with:

* `docs/overview.md`
* `docs/source-inventory.md`
* `docs/event-schema.md`
* `docs/sample-dataset.md`
* `docs/exploratory-report.md`
* `docs/baseline-targets.md`
* `docs/next-actions.md`

Update `docs/next-actions.md` whenever completing or changing the immediate work queue.

## Data Rules

* Limit project data to Italy.
* Store raw source records unchanged before normalization.
* Normalize timestamps to UTC.
* Preserve source identifiers, source URIs, and uncertainty fields.
* Keep source connectors separate from normalization, feature generation, modeling, and evaluation.

## Source Validation

Mark a source usable only after a reproducible sample pull works.

For INGV, prefer the public FDSN event service documented in `docs/source-inventory.md`. Some query variants may fail; record the exact working request shape before relying on it.

## Modeling Rules

Start with naive and historical-rate baselines. Do not add complex ML until event catalog access, normalized schema, target definition, and validation split are stable.

Use time-based validation first. Training data must occur before validation data.
