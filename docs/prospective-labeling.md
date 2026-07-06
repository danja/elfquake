# Prospective Labeling

First VLF-anchored rows become labelable after:

`2026-07-06T09:57:24Z`

Completed pre-label refresh:

* INGV fetched for `2026-06-29T00:00:00Z` through `2026-07-07T00:00:00Z` at `2026-07-06T10:38:02Z`.
* Combined event tables:
  * `data/derived/ingv/events_italy_2026-06-01_2026-07-07.combined.normalized.csv`
  * `data/derived/ingv/events_central_italy_2026-06-01_2026-07-07.combined.normalized.csv`
* Rebuilt prospective image-window tables:
  * `data/derived/multimodal/central_italy.prospective_vlf_image_windows.csv`
  * `data/derived/multimodal/all_italy.prospective_vlf_image_windows.csv`

Current status as of `2026-07-06T12:36:24Z`:

* 247 rows per region
* combined events: 174 all-Italy, 21 central-Italy
* no missing VLF, VLF image, or astronomy coverage
* first target window labeled
* central Italy: 1 labeled negative, 0 labeled positive
* all Italy: 0 labeled negative, 1 labeled positive
* no additional rows are ready to label yet
* model readiness remains `waiting_for_labels`

Label outputs:

* `data/derived/multimodal/central_italy.prospective_vlf_image_windows.labeled.csv`
* `data/derived/multimodal/all_italy.prospective_vlf_image_windows.labeled.csv`

The first labels were produced with:

```sh
PYTHONPATH=src python -m elfquake.cli label-multimodal-targets --input data/derived/multimodal/central_italy.prospective_vlf_image_windows.csv --events data/derived/ingv/events_central_italy_2026-06-01_2026-07-07.combined.normalized.csv --as-of 2026-07-06T10:38:02Z --out data/derived/multimodal/central_italy.prospective_vlf_image_windows.labeled.csv
PYTHONPATH=src python -m elfquake.cli label-multimodal-targets --input data/derived/multimodal/all_italy.prospective_vlf_image_windows.csv --events data/derived/ingv/events_italy_2026-06-01_2026-07-07.combined.normalized.csv --as-of 2026-07-06T10:38:02Z --out data/derived/multimodal/all_italy.prospective_vlf_image_windows.labeled.csv
```

Continue refreshing and labeling as more target windows mature. Only train smoke models after both target classes exist in the same labeled table.
