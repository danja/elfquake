# Roadmap

ELFQuake should move from data feasibility to modeling only after each phase has reproducible outputs.

## Phases

1. Data connection: confirm access to seismic, radio, and astronomical sources.
2. Data analysis: inspect coverage, gaps, noise, bias, and timestamp alignment.
3. Simulation construction: create simple replay and backtesting workflows.
4. Model construction: start with naive and classical baselines before complex ML.
5. Model evaluation: compare against held-out time and geography splits.
6. End-to-end system: connect ingestion, feature generation, training, and reports.
7. System evaluation: verify repeatability, drift, latency, and operational costs.
8. Deployment: publish only validated analysis outputs and documented limitations.

## Milestone Order

* Build a reproducible data inventory.
* Create a normalized event and signal dataset.
* Produce exploratory reports.
* Define baseline targets and metrics.
* Run backtests before adding advanced models.
