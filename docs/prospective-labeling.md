# Prospective Labeling

First VLF-anchored rows become labelable after:

`2026-07-06T09:57:24Z`

Before labeling:

1. Fetch INGV events covering the target period, at least `2026-06-29` through `2026-07-07`.
2. Normalize all-Italy and Central Italy event tables.
3. Combine new normalized events with the existing June table.
4. Run `label-multimodal-targets` on:
   * `data/derived/multimodal/central_italy.prospective_vlf_image_windows.csv`
   * `data/derived/multimodal/all_italy.prospective_vlf_image_windows.csv`
5. Re-run `summarize-prospective-table` and only train smoke models if both target classes exist.

Current status:

* 12 rows per region
* no missing VLF, VLF image, or astronomy coverage
* no rows ready to label on `2026-06-29`
