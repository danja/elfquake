# Status

ELFQuake is currently a reproducible research pipeline rather than a validated prediction system.

The Italy track has INGV seismic acquisition, live Cumiana VLF capture, astronomical and space-weather connectors, synthetic avalanche data, CPU PyTorch baselines, and chronological holdout experiments. The Japan track has research-only ISEE Moshiri CDF ingestion and a recent exploratory earthquake/VLF window analysis.

The main limitation is data coverage: real VLF observations do not yet provide enough well-aligned positive and negative seismic outcomes for a fair supervised multimodal test. Results are therefore reported against seismic-only baselines, missing-modality controls, and held-out time periods.

The next priorities are longer VLF coverage, matched controls around major events, broader synthetic regimes, and repeated cross-region transfer experiments. See [next actions](next-actions.md) and the [latest report](report.md).
