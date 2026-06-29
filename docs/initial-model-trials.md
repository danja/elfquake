# Initial Model Trials

Command:

`train-ablation-smoke`

Current reports:

* `data/derived/multimodal/central_italy_2026-06-01_2026-06-29.ablation_smoke.json`
* `data/derived/multimodal/all_italy_2026-06-01_2026-06-29.ablation_smoke.json`
* `data/derived/multimodal/central_italy.prospective_vlf_image_windows.ablation_smoke.json`

Results:

* Central Italy June rows train in-sample for seismic-only, seismic+astronomy, seismic+VLF, and full ablations.
* All Central Italy ablations score `1.0` in-sample accuracy on only three rows. This is a pipeline check, not evidence.
* All-Italy June rows are not trainable because all three targets are positive.
* Prospective VLF rows are not trainable yet because labels are pending.

Current interpretation:

* seismic-only already fits the tiny Central Italy sample, so VLF value cannot be evaluated on this data.
* historical VLF image features are unavailable for the June labeled rows.
* the first meaningful VLF ablation requires labeled prospective rows after `2026-07-06`.
