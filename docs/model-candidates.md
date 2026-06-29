# Model Candidates

Start with models that can prove whether VLF and astronomy add value over seismic-only features.

## First Models

1. Historical-rate baseline by region and time window.
2. Regularized logistic regression on tabular window features.
3. Gradient-boosted trees on tabular multimodal features.

The first model table should be assembled from labeled windows, starting with [Training Windows](training-windows.md).

## Later Models

Use temporal models only after enough aligned windows exist:

* temporal convolution or GRU over window sequences
* late-fusion model combining seismic, VLF image summaries, and astronomy features
* transformer-style sequence model only if the dataset becomes large enough

## Evaluation Rule

Every candidate must be compared through ablations:

* seismic only
* seismic plus VLF
* seismic plus astronomy
* seismic plus VLF and astronomy

Do not optimize model complexity before source coverage and target labels are reliable.
