# Design Matrix

Current model-shaped tables:

* `data/derived/multimodal/central_italy_2026-06-01_2026-06-29.design_matrix.csv`
* `data/derived/multimodal/all_italy_2026-06-01_2026-06-29.design_matrix.csv`

Inputs:

* labeled seismic window table
* corrected GFZ Kp/Ap normalization
* corrected Space Weather Canada F10.7 normalization

Current row includes:

* seismic event count and max magnitude
* Kp mean/max
* ap mean/max
* F10.7 mean
* source coverage counts
* target label

Model smoke reports:

* `data/derived/multimodal/central_italy_2026-06-01_2026-06-29.logistic_smoke.json`
* `data/derived/multimodal/all_italy_2026-06-01_2026-06-29.logistic_smoke.json`

These are pipeline checks only. The Central Italy report fits three rows in-sample; all-Italy currently lacks class variation.
