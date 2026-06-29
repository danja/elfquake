# Next Actions

Prioritize feasibility and reproducibility before modeling.

1. Confirm archival access for NOAA geomagnetic and solar data before backtesting.
2. Run acquisition smoke captures for INGV, Cumiana VLF, and astronomy sources.
3. Convert the manual seismic normalization into repeatable code.
4. Expand source access beyond the smoke window.
5. Create the first multimodal smoke input row from available source manifests.

Completed:

* Create an Italy-scoped data source inventory.
* Capture initial licensing, access methods, formats, and cadence.
* Define the first normalized event schema.
* Define the first sample dataset.
* Define the exploratory report template.
* Define baseline prediction targets and metrics.
* Record initial working INGV connector notes.
* Build the first Italy smoke dataset from the working INGV text export.
* Define the local normalization step for the INGV text export.
* Create the normalized smoke dataset from the INGV text export.
* Derive a Central Italy subset from the normalized smoke dataset.
* Produce the first exploratory report.
* Define the naive historical-rate baseline input table.
* Run a naive historical-rate baseline smoke output before adding ML models.
* Evaluate initial vlf.it availability, licensing signal, and data gaps.
* Confirm Cumiana live VLF spectrogram and plot image endpoints.
* Define the Cumiana VLF image capture manifest and storage layout.
* Evaluate astronomical and geomagnetic data availability for the same windows.
* Define the multimodal time-window schema joining seismic, VLF, and astronomical data.
* Scaffold stdlib-only raw data acquisition and storage code.
