# Multimodal Smoke Row

First generated row:

`data/derived/multimodal/central_italy_2026-06-22_2026-06-29T1015.multimodal_smoke.csv`

Purpose:

* prove seismic, VLF, and astronomy captures can be aligned into one table
* expose missing or stale source coverage as fields
* keep target fields unlabeled until the future target window is available

Current inputs:

* seismic: Central Italy INGV events, 4 events in the feature window
* VLF: Cumiana `last_E_VLF` image captured before `target_start_utc`
* astronomy: USNO moon phases and NOAA monthly F10.7

The row is not model-training data yet. It is a schema and provenance smoke artifact.
