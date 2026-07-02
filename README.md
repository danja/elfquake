# ELFQuake

ELFQuake is a research project testing whether VLF radio observations can augment seismic and astronomical data enough to support useful earthquake-related predictive modeling.

The core hypothesis is that natural ELF/VLF radio anomalies may contain signal that is not present in seismic event history alone. *To be useful this claim must be demonstrated against reproducible seismic-only baselines, held-out time periods, and multimodal ablations.*

### Status

Data acquisition, feature extraction, prospective rows, smoke models, CPU-only sandpile simulation, and first synthetic VLF comparison tooling are implemented; **no earthquake prediction capability is claimed**.

Right now, while awaiting further data from INGV, the focus is on the simulation. (The first prospective VLF rows become labelable after `2026-07-06T09:57:24Z`).

## Current Work

* INGV seismic event acquisition, normalization, and Italy/Central Italy training windows.
* Cumiana VLF live spectrogram capture through systemd, with pixel-derived image features.
* Astronomical and space-weather archive connectors and normalization.
* Prospective VLF-anchored feature rows with pending target labels.
* Dependency-light logistic and ablation smoke models for feasibility checks.
* Sandpile simulation with seismic-like avalanche outputs, pre-avalanche piezo strain sensors, and avalanche-derived piezo/VLF analogue outputs.

## Simulation

The simulation is an artificial mountain-like grid where broad background loading is combined with repeated localized stress at fixed point sources. As slopes become unstable, small avalanches redistribute height to neighbouring cells. The aim is to generate synthetic sequences that are close enough in shape to real-world seismic data to be useful as training data for a deep learning system, especially before enough matched seismic, VLF, and astronomical data is available.

It also includes piezo-like sensors. The pre-avalanche channel watches quartz-like susceptible regions near failure, while the default VLF/WAV demo uses an avalanche-derived channel computed from actual toppling and stress release so spikes come from the simulation dynamics rather than display-time noise.

Run the local simulation demo pipeline with:

```sh
./run-all.sh
```

Default outputs use `data/derived/sim/mountain_256x256_seed42_10000` as the prefix. The normal piezo image is `*.piezo_vlf_summary.png`; the older FFT diagnostic is opt-in with `RUN_FFT=1`.

Compare the simulated VLF analogue image against captured Cumiana VLF spectrograms with:

```sh
./compare-piezo-vlf.sh
```

This is a simplified stress-and-release analogy, not a geological model. Its value depends on whether the generated data has useful structural similarity to real observations. Good performance on simulated avalanches would only show that the tooling can learn synthetic patterns; real claims still require held-out seismic, VLF, and astronomical data.

## Key Docs

* [Overview](docs/overview.md)
* [Next Actions](docs/next-actions.md)
* [Development Environment](docs/development-environment.md)
* [Source Inventory](docs/source-inventory.md)
* [Multimodal Feasibility](docs/multimodal-feasibility.md)
* [Systemd Service](docs/systemd-service.md)
* [Initial Model Trials](docs/initial-model-trials.md)
* [Sandpile Simulation](docs/sandpile-simulation.md)
