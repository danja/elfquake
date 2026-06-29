# Training Windows

Initial labeled training-window builder:

`build-seismic-training-windows`

Current output:

`data/derived/multimodal/central_italy_2026-06-01_2026-06-15.seismic_training_windows.csv`

Current row:

* region: `central_italy`
* feature window: `2026-06-01T00:00:00Z` to `2026-06-08T00:00:00Z`
* target window: `2026-06-08T00:00:00Z` to `2026-06-15T00:00:00Z`
* target: M3.0+
* label: `target_occurred=0`

This is a seismic-only seed table. Join VLF and archive-backed astronomy features only after matching historical source coverage exists.
