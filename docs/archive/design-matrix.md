# Design Matrix

Current model-shaped tables:

* `data/derived/multimodal/central_italy_2026-06-01_2026-06-29.design_matrix.csv`
* `data/derived/multimodal/all_italy_2026-06-01_2026-06-29.design_matrix.csv`
* `data/derived/multimodal/central_italy_2026-06-01_2026-06-29.multimodal_design_matrix.csv`
* `data/derived/multimodal/all_italy_2026-06-01_2026-06-29.multimodal_design_matrix.csv`

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
* VLF capture counts and quality flags in the multimodal variant

Model smoke reports:

* `data/derived/multimodal/central_italy_2026-06-01_2026-06-29.logistic_smoke.json`
* `data/derived/multimodal/all_italy_2026-06-01_2026-06-29.logistic_smoke.json`
* `data/derived/multimodal/central_italy_2026-06-01_2026-06-29.multimodal_logistic_smoke.json`
* `data/derived/multimodal/all_italy_2026-06-01_2026-06-29.multimodal_logistic_smoke.json`

These are pipeline checks only. The June VLF columns are missing for the labeled windows because the live service started after them.
