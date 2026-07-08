# Model Readiness

Use `summarize-model-readiness` before training.

It checks:

* labeled vs unlabeled row counts
* positive and negative target counts
* available feature groups: seismic, astronomy, VLF metadata, VLF image
* whether planned ablations have the required feature columns

Current reports:

* `data/derived/multimodal/central_italy.prospective_vlf_image_windows.readiness.json`
* `data/derived/multimodal/all_italy.prospective_vlf_image_windows.readiness.json`
* `data/derived/multimodal/central_italy_2026-06-01_2026-06-29.multimodal_readiness.json`
* `data/derived/multimodal/all_italy_2026-06-01_2026-06-29.multimodal_readiness.json`

Current status:

* prospective VLF rows have all ablation feature groups but are waiting for labels
* Central Italy June smoke rows have both classes but no historical VLF image features
* all-Italy June smoke rows lack class variation
* current synthetic smoke modeling should use the hourly `gt0` target described in [Model Targets](model-targets.md)
