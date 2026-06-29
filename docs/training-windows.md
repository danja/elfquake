# Training Windows

Initial labeled training-window builder:

`build-seismic-training-windows`

Current expanded outputs:

* `data/derived/multimodal/central_italy_2026-06-01_2026-06-29.seismic_training_windows.csv`
* `data/derived/multimodal/all_italy_2026-06-01_2026-06-29.seismic_training_windows.csv`

Current shape:

* 7-day feature windows
* 7-day target windows
* target: M3.0+
* three labeled June rows per table

Central Italy has both classes in this tiny sample. All-Italy is all positive at M3.0+, so it needs a different threshold, regioning, or more history.
