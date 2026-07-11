# ELFQuake

#### [Questa pagina in italiano](README.it.md)

ELFQuake is a research project examining whether [Extremely](https://en.wikipedia.org/wiki/Extremely_low_frequency)/[Very Low Frequency](https://en.wikipedia.org/wiki/Very_low_frequency) radio observations can augment seismic and astronomical data enough to support useful predictive modeling of earthquakes.

The core hypothesis is that natural ELF/VLF radio anomalies may contain signal that is not present in seismic event history alone. At least [one study](https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/113/6/2461/627949/Earthquake-Forecasting-Using-Big-Data-and)
 suggests this may be a viable approach.

We wish to exploit more modern machine learning/AI techniques to create a predictive model based on the transformer architecture. A key consideration is the availability of real-world data for training. The plan is to build a system (based around an avalanche model) with similar chacteristics to the geological system and use this to generate synthetic data for baseline training. After validation of this stage, real-world data will be used for fine-tuning.

*To be useful any claims must be demonstrated against reproducible seismic-only baselines, held-out time periods, and multimodal ablations.*

## Status

Status: the end-to-end trial emits next-week `>M2` event coordinates, while live labels remain one-class and synthetic causal precursor checks are still unstable; **no earthquake prediction capability is claimed**.

Right now, while awaiting further VLF-aligned labels, the focus is on label-free real VLF representation learning, real seismic baselines, synthetic regime diagnostics, and keeping the multimodal model interface stable. Evaluation of the current model can be found in [report.md](docs/report.md), [model-comparison.md](docs/model-comparison.md), and [model-scaling-requirements.md](docs/model-scaling-requirements.md).

### Simulated Earthquakes

![simulated earthquake map](docs/images/map_v1.1.0.png)

### Simulated VLF Signal

![simulated VLF emissions](docs/images/vlf-simulated.png)


## Background

This work was initially prompted by the tragedy of the [2009 L'Aquila earthquake](https://en.wikipedia.org/wiki/2009_L'Aquila_earthquake). Around the same period I was aware of developments in Deep Learning and coincidentally had stumbled on material related to natural radio signals occurring as precursors to seismic events (see [vlf.it](http://www.vlf.it/)). I made a start on research and began a blog about it : [ELFQuake](https://elfquake.wordpress.com/) (which I wound up using for general blogging). At the time it seemed possible but *very difficult*. But since then predictive models have improved in leaps and bounds, meaning that prediction has a much better chance of working. Not only that, but the ability to delegate much of the coding work to an intelligent assistant means that the difficulty in building the system has been drastically reduced. 

## Current Work

* INGV seismic event acquisition, normalization, and Italy/Central Italy training windows, currently backfilled from 2024-01-01 through 2026-07-07.
* Cumiana VLF live spectrogram capture through systemd, with pixel-derived image features.
* Astronomical and space-weather archive connectors and normalization.
* Prospective VLF-anchored feature rows with pending target labels.
* A CPU PyTorch self-supervised autoencoder over real Cumiana VLF image sequences.
* A modality-aware masked-patch Transformer evaluation comparing synthetic, real VLF, sequential-transfer, joint, and random initialization over repeated seeds.
* Dependency-light logistic checks and a CPU PyTorch tabular MLP for synthetic aligned rows.
* A first CPU PyTorch GRU sequence model over synthetic avalanche and piezo/VLF sensor tensors.
* A CPU PyTorch patch Transformer scaffold for synthetic sequence engineering checks and event-list target pretraining.
* Summary comparison, sequence sweep, and missing-modality scripts for model diagnostics.
* A compact real-vs-synthetic comparison wrapper for central-Italy seismic baselines and synthetic sequence reports.
* Larger-model scale checks for row count, class balance, sequence size, and CPU-only model limits.
* Real Cumiana VLF image features materialized into the same sequence shape as synthetic piezo/VLF inputs.
* Real prospective aligned model inputs scaffolded for all-Italy and central-Italy rows; real deep-model fine-tuning wrappers refuse to train until both target classes exist.
* VLF model feature roles that allow synthetic piezo/VLF analogue data to exercise the same PyTorch VLF path before real labels mature.
* Synthetic-inlier transfer diagnostics that train on real-like synthetic piezo/VLF descriptors and evaluate reconstruction against held-out real VLF descriptors.
* Mixed real/synthetic VLF descriptor alignment with CORAL loss and centroid/random/full synthetic controls.
* Label-free real VLF anomaly smoke forecasts while supervised real labels remain blocked.
* A deterministic trial weekly event-list forecast that combines historical INGV rates/locations, real VLF context, astronomy context, and synthetic avalanche spatial priors into a downstream-ready CSV/JSON contract.
* A first swappable learned scorer trained on synthetic aligned rows, with the same weekly event-list CSV contract and historical INGV rate calibration metadata.
* Synthetic event-list target and model heads for occurrence, count, magnitude, and centroid, currently useful as an engineering adapter but not yet temporally robust.
* A Transformer target adapter and sweep wrappers for training against the richer h6 synthetic event-list target table.
* Drift-aware synthetic validation and shorter-episode simulation scaffolding to reduce one-run lifecycle bias; the first aggressive 3000-step profile fixes target-rate drift in a small probe.
* A larger warmed synthetic batch wrapper for CPU-only Transformer pretraining data expansion.
* Sandpile simulation with separate seismic-like avalanche outputs and piezo/VLF analogue outputs.

Run the default label-free real VLF pretraining path with:

```sh
./scripts/pretrain-real-vlf-self-supervised.sh
```

Compare self-supervised Transformer initialization strategies with:

```sh
./scripts/evaluate-self-supervised-transformer.sh
```

Run the current label-free 7-day VLF anomaly smoke forecast with:

```sh
./scripts/score-real-vlf-anomaly-forecast.sh
```

Run the current end-to-end trial `>M2` weekly event-list forecast with:

```sh
./scripts/trial-weekly-event-forecast.sh
```

Run the synthetic-trained learned weekly event-list forecast with:

```sh
./scripts/learned-weekly-event-forecast.sh
```

Run the current synthetic event-list patch Transformer smoke with:

```sh
./scripts/train-synthetic-event-list-patch-transformer.sh
```

Compare the current real VLF embedding domain against synthetic piezo/VLF analogues with:

```sh
./scripts/compare-vlf-embedding-domains.sh
```

Run the synthetic-inlier transfer diagnostic with:

```sh
./scripts/evaluate-vlf-synthetic-inlier-transfer.sh
```

Run the mixed-domain VLF alignment diagnostic with:

```sh
./scripts/evaluate-vlf-mixed-domain-alignment.sh
```

## Simulation

The simulation is an artificial mountain-like grid where broad background loading is combined with repeated localized stress at fixed point sources. As slopes become unstable, small avalanches redistribute height to neighbouring cells. The aim is to generate synthetic sequences that are close enough in shape to real-world seismic data to be useful as training data for a deep learning system, especially before enough matched seismic, VLF, and astronomical data is available.

It also includes piezo-like sensors that watch quartz-like susceptible regions near failure and produce the VLF/WAV analogue channel. Direct seismic-like event data is kept separate and is derived from avalanche/toppling behavior.

Run the local simulation demo pipeline with:

```sh
./scripts/run-all.sh
```

Default outputs use `data/derived/sim/mountain_256x256_seed42_10000` as the prefix. The normal piezo image is `*.piezo_vlf_summary.png` from `*.piezo.csv`; the direct seismic event analogue is `*.avalanche_events.csv`. The older FFT diagnostic is opt-in with `RUN_FFT=1`.

The event-map demo projects avalanche-derived locations over an Apennine-style Italy belt and uses point size for synthetic magnitude.

Render a demo overlay of actual synthetic avalanche events and PyTorch predicted-positive target-window hits with:

```sh
./scripts/prediction-event-map.sh
```

Compare the simulated VLF analogue image against captured Cumiana VLF spectrograms with:

```sh
./scripts/compare-piezo-vlf.sh
```

This is a simplified stress-and-release analogy, not a geological model. Its value depends on whether the generated data has useful structural similarity to real observations. Good performance on simulated avalanches would only show that the tooling can learn synthetic patterns; real claims still require held-out seismic, VLF, and astronomical data.

## Key Docs

* [Overview](docs/overview.md)
* [Documentation Index](docs/README.md)
* [Processing Graph](docs/processing-graph.md)
* [Next Actions](docs/next-actions.md)
* [Forecast Interface](docs/forecast-interface.md)
* [Success Criteria](docs/success-criteria.md)
* [Development Environment](docs/development-environment.md)
* [Source Inventory](docs/source-inventory.md)
* [Multimodal Feasibility](docs/multimodal-feasibility.md)
* [Systemd Service](docs/systemd-service.md)
* [Initial Model Trials](docs/initial-model-trials.md)
* [Model Comparison](docs/model-comparison.md)
* [Model Scaling Requirements](docs/model-scaling-requirements.md)
* [Sandpile Simulation](docs/sandpile-simulation.md)
