# Next Actions

Prioritize feasibility and reproducibility before modeling.

1. Reinstall/reload the updated prospective systemd unit if timer-managed image features and summaries are desired.
2. Keep the VLF capture and prospective timers running until the first target windows mature.
3. On or after `2026-07-06T09:57:24Z`, refresh INGV events through `2026-07-07` and label the first prospective rows.
4. Add chunked sandpile snapshot storage only if larger pretraining runs outgrow `.npy` sanity snapshots.
5. Add slope/erosion smoothing to mountain-mode synthetic terrain if ridgeline-like visuals are needed.
6. Backfill enough historical INGV windows to get both positive and negative target classes by region.
7. Continue with prospective-only VLF evaluation unless a separate historical Cumiana archive is obtained.
8. Replace the schematic event-map basemap with a real cartographic basemap if geospatial dependencies are installed.
9. Add avalanche centroid or rupture-mask outputs so synthetic event maps are not limited to fixed sensor proxy locations.
10. Define regular-cadence multimodal tensors with missing-data masks before implementing Transformer candidates.

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
* Refresh current prospective VLF image feature tables and summaries.
* Check the Cumiana live HTML page for archive links; none were exposed.
* Add a concise first-rollover labeling runbook.
* Add model-readiness reports for label balance and ablation feature groups.
* Run initial in-sample ablation smoke models on currently labeled tables.
* Add CPU-only sandpile simulation CSV output for deterministic synthetic avalanche sequences.
* Add sandpile JSON summary and CPU benchmark smoke reports.
* Add optional sandpile `.npy` grid snapshots and PNG heatmap rendering for sanity checks.
* Add periodic sandpile CLI progress output and batch heatmap rendering from captured snapshots.
* Add corrective sandpile safety draining so z-axis overflow is stabilized and recorded.
* Add an ffmpeg helper script for building MP4 videos from sandpile heatmap PNGs.
* Add sandpile mountain mode with target mean height and periodic bottom-layer removal.
* Add fixed-scale sandpile heatmap rendering so colors reflect absolute height across frames.
* Change sandpile relaxation to topple local neighbour slopes rather than absolute z-height.
* Add sandpile heatmap render progress, CPU worker parallelism, and broader color controls.
* Add a dry-run derived-data cleanup helper.
* Slow mountain-mode target filling and scale deposition defaults so videos visibly develop.
* Restore localized point-source stress deposition as the mountain-mode default.
* Add a modular piezo-like precursor sensor channel for pre-avalanche stress signals.
* Add derived synthetic INGV-like event lists and piezo spectrogram rendering.
* Correct piezo spectrogram frequency axis to derive from simulation timestep.
* Add combined piezo time-series/spectrogram PNG and WAV sonification helpers.
* Compress long piezo plots for display and smooth WAV sonification defaults.
* Add single-sensor piezo rendering and DC-blocking filters for diagnostics.
* Replace instantaneous piezo emission with a stateful charge-store/release model.
* Add an offline Italy event-map renderer for normalized INGV-like event CSVs.
* Update model candidates with time-series Transformer architectures from `2202.07125v5`.
