# VLF Coverage

Current commands:

* `build-vlf-window-features`
* `join-vlf-design-matrix`
* `build-prospective-vlf-windows`
* `extract-vlf-image-features`
* `record-vlf-abelian-cumiana`
* `fetch-vlf-abelian-cumiana-archive`
* `extract-vlf-audio-features`

The June training windows currently have `quality_missing_vlf=1` because Cumiana service captures began after the labeled windows. Keep VLF columns in the matrix, but evaluate VLF value only on windows with real aligned captures.

The vlf.it Cumiana page exposes current image endpoints only. Abelian now provides a Cumiana `vlf15` live Ogg stream and a short-window archive retrieval form; use it for live or historical work only after a reproducible nonempty pull is confirmed.

Prospective service rows:

* `data/derived/multimodal/central_italy_2026-06-29.prospective_vlf_windows.csv`
* `data/derived/multimodal/all_italy_2026-06-29.prospective_vlf_windows.csv`
* `data/derived/multimodal/central_italy.prospective_vlf_windows.csv`
* `data/derived/multimodal/all_italy.prospective_vlf_windows.csv`

These rows are VLF-anchored, unlabeled, and targetable after their `target_end_utc`.

Use `update-prospective-vlf-table` for recurring updates. It preserves existing rows and appends only unseen `window_id`s.

Image-derived feature tables:

* `data/derived/multimodal/cumiana_last_E_VLF_2026-06-29.image_features.csv`
* `data/derived/multimodal/cumiana_other_vlf_2026-06-29.image_features.csv`
* `data/derived/multimodal/cumiana_last_E_VLF.image_features.csv`
* `data/derived/multimodal/central_italy.prospective_vlf_image_windows.csv`
* `data/derived/multimodal/all_italy.prospective_vlf_image_windows.csv`
* `data/derived/multimodal/central_italy.prospective_vlf_image_windows.summary.json`
* `data/derived/multimodal/all_italy.prospective_vlf_image_windows.summary.json`

These summarize cropped JPEG spectrogram pixels: intensity distribution, hot-color ratio, band means, and vertical streak count. They are proxy features until raw VLF samples are available.

Use `summarize-prospective-table` to check row counts, missing source coverage, and rows ready for target labeling.

See [Prospective Labeling](prospective-labeling.md) for the first label rollover.
