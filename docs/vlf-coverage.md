# VLF Coverage

Current commands:

* `build-vlf-window-features`
* `join-vlf-design-matrix`
* `build-prospective-vlf-windows`

The June training windows currently have `quality_missing_vlf=1` because Cumiana service captures began after the labeled windows. Keep VLF columns in the matrix, but evaluate VLF value only on windows with real aligned captures.

Prospective service rows:

* `data/derived/multimodal/central_italy_2026-06-29.prospective_vlf_windows.csv`
* `data/derived/multimodal/all_italy_2026-06-29.prospective_vlf_windows.csv`

These rows are VLF-anchored, unlabeled, and targetable after their `target_end_utc`.
