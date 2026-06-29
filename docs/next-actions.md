# Next Actions

Prioritize feasibility and reproducibility before modeling.

1. Run planned INGV live backfill windows when network use is intended.
2. Install and smoke-test the systemd service on the target host.
3. Run a live Cumiana cadence check when a 30-minute wait is acceptable.
4. Backfill enough historical INGV windows to get both positive and negative target classes by region.
5. Determine whether usable historical Cumiana VLF imagery exists; otherwise plan prospective-only VLF evaluation.
6. Install the prospective timer if recurring row generation should run under systemd.
7. Refresh the normalized INGV event table before labeling future prospective rows.
8. After `2026-07-06`, label the first prospective VLF rows against INGV events.
9. Reinstall/reload the updated prospective systemd unit if timer-managed image features are desired.

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
* Run acquisition smoke captures for INGV, Cumiana VLF, and astronomy sources.
* Add acquisition command error handling and source-level smoke tests.
* Convert the manual seismic normalization into repeatable code.
* Check official NOAA SWPC archive feasibility for geomagnetic and solar backtesting.
* Find non-rolling archive sources for Kp, Dst, GOES X-ray, and daily F10.7.
* Add archive connector stubs for GFZ Kp/Ap, Kyoto Dst, NCEI GOES XRS, and daily F10.7.
* Partition live endpoint tests from offline mock tests.
* Expand INGV source access to a 14-day live Italy window and normalized Central Italy subset.
* Create the first multimodal smoke row from seismic, VLF, and astronomy captures.
* Add a polite repeated-capture runner for Cumiana VLF cadence checks.
* Add coarse VLF and astronomy feature extraction stubs.
* Add archive normalization stubs for Kp/Ap, Dst, GOES XRS, and F10.7.
* Split longer INGV pulls into service-friendly windows before backfilling.
* Add target-label generation for elapsed multimodal windows.
* Add a manifest-driven feature-table builder across many windows.
* Add no-dependency VLF image summaries for dimensions and byte entropy.
* Add NetCDF-backed GOES XRS archive normalization.
* Run a live batch across INGV, Cumiana VLF, NOAA/USNO, GFZ, F10.7, and NCEI GOES.
* Define first model candidates for multimodal tabular and temporal data.
* Add time-windowed GOES XRS normalization for routine batches.
* Build first labeled seismic training window table.
* Join archive-backed Kp/Ap and F10.7 features into a first design matrix.
* Verify service-produced VLF capture through a derived feature row.
* Combine June normalized INGV event segments into Central Italy and all-Italy tables.
* Expand June labeled training windows to three 7-day feature/target rows.
* Add a dependency-free logistic regression smoke trainer and first model reports.
* Add VLF window feature generation from service capture metadata.
* Join VLF coverage columns into the multimodal design matrix with explicit missing flags.
* Add prospective VLF-anchored rows with pending target labels.
* Add idempotent cumulative prospective VLF table updates.
* Add a systemd timer/service template for prospective row generation.
* Add pixel-derived VLF image feature extraction from Cumiana JPEGs.
* Join image-derived VLF features into prospective rows by capture timestamp.
* Add prospective table quality summaries for coverage and label readiness.
