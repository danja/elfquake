# ELFQuake: A Working Experiment, Not Yet a Predictor

*15 July 2026*

ELFQuake is a research project asking a difficult question: can earthquake records be combined with very-low-frequency radio observations and astronomical data to improve forecasts of earthquakes in Italy?

The important word is "improve". A model that appears accurate but does no better than a simple historical estimate is not useful. The project therefore treats every result as an experiment against clear baselines, with the future held out from training.

## The System Is Now End to End

The project can now collect and organize the main data sources, train CPU-based models, and produce a seven-day map of actual and predicted earthquake locations.

The real earthquake catalogue comes from INGV, Italy’s national earthquake data service. The current historical file contains 4,836 events from January 2024 to July 2026. Cumiana VLF observations are being collected as spectrogram images. A spectrogram is simply a picture showing how radio energy changes over time and across frequencies. Astronomy and space-weather connectors are also in place.

The VLF record is not yet long enough to support a fair supervised test. The available earthquake-aligned VLF rows still contain only one class of outcome in the current windows. In plain terms, the system has examples of one kind of week, but not enough examples of both "an event followed" and "no event followed". The model therefore cannot honestly learn or test a real VLF earthquake signal yet.

## Why Use a Sandpile Simulation?

Real strong earthquakes are rare, and the suspected radio effects are even harder to label. ELFQuake uses a sandpile-style avalanche simulation as a controlled source of synthetic data.

The simulation represents stress building up at localized points, followed by sudden avalanches. It produces two separate signal families:

- a direct avalanche signal, intended as a rough analogue of seismic activity;
- a piezo-like signal, intended as a rough analogue of a radio response caused by stress in quartz-bearing rock.

These signals are not claimed to be physically equivalent to earthquakes or VLF radio. Their purpose is to provide test data for the software and model interfaces while the real record grows. The simulation is also useful for checking whether an apparently promising model works across different runs, rather than only on one convenient example.

A new 20,000-step CPU simulation run was completed at seed 4300. Together with three earlier long runs, the dense synthetic corpus now contains 79,976 records. Independent simulations share a demonstration start date, so the training pipeline offsets their synthetic times when stacking them. This changes no real timestamp; it only prevents separate experiments from being mistaken for simultaneous observations.

## The First Synthetic-to-Real Test

An all-Italy weekly target is too easy to define badly: at a threshold around magnitude 2.5, Italy has an event in most weeks. To create a meaningful comparison, the country was divided into fixed geographic cells. The task became:

> Will a particular cell contain at least one magnitude 2.5 or greater earthquake during the next seven days?

The model was trained on the first 80% of the real timeline and evaluated on the final 20%. This is a chronological holdout: the model only sees the past, then is tested on a later period.

Three approaches were compared:

- a historical spatial-rate baseline, which predicts using how often each cell has produced events in the past;
- a small neural network trained only on real seismic features;
- the same type of network first trained on synthetic data, then fine-tuned (adapted) using real seismic data.

The historical baseline achieved a balanced accuracy of `0.686` and precision of `0.344`. Balanced accuracy gives equal weight to finding event cells and correctly rejecting quiet cells. Precision measures how many predicted cells really contained an event.

The real-only model reached `0.666` balanced accuracy and `0.269` precision. Synthetic pretraining improved this to `0.681` and `0.307`, but it still did not beat the historical baseline. Four rolling tests, using several different training cutoffs, produced an average balanced accuracy of `0.668`, with the weakest fold at `0.656`.

These results are useful because they show the complete process working, but they do not show useful earthquake prediction. The model is not yet adding reliable information beyond the known geographical distribution of Italian seismicity.

## A Map for Inspection

The system also renders a randomly selected held-out week on an offline map of Italy. Blue circles show actual INGV events; red crosses show the centres of cells predicted by the model. Marker size represents the magnitude proxy.

The map is deliberately a diagnostic picture, not a forecast product. Predicted crosses are independently generated cell locations, not actual events relabelled after the fact.

## What Comes Next

The immediate work is to grow the synthetic corpus with more independent long runs and repeat the transfer experiment. The aim is to establish whether synthetic pretraining consistently helps across time periods, not merely in one split.

The real model will also be tested at several cell sizes and magnitude thresholds, with the choice made using training data only. More rolling holdouts will show whether performance survives changes in time period.

Most importantly, VLF data will remain a separate ablation (a controlled test with one input removed). Once the live record contains both positive and negative outcomes, seismic-only, seismic-plus-VLF, and astronomy-augmented models can be compared fairly.

At this point ELFQuake has a reproducible research pipeline and a clear baseline. It does not have demonstrated earthquake prediction capability. That is a useful result: the next improvements can now be judged against a real held-out standard rather than against a demonstration that merely looks convincing.
