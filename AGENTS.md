# AGENTS.md

Guidance for agents working in this repository.

## Project Scope

ELFQuake is an Italy-scoped research project for testing whether seismic, VLF radio, and astronomical data contain useful earthquake-related signals.

The central hypothesis is that VLF radio data may augment seismic data enough to make useful predictive models possible.

Do not claim earthquake prediction capability unless it is demonstrated against reproducible baselines and held-out data.

## Current Priority

Prioritize feasibility and reproducibility:

1. Verify VLF and astronomical data access, licensing, timestamps, and station/source metadata.
2. Keep the seismic smoke pipeline reproducible.
3. Define multimodal time-window alignment.
4. Compare seismic-only baselines against multimodal variants.
5. Use ablations to measure whether VLF or astronomical features add value.

## Documentation

Keep docs concise and modular. Prefer one document per concern under `docs/`.

Start with:

* `docs/overview.md`
* `docs/source-inventory.md`
* `docs/multimodal-feasibility.md`
* `docs/event-schema.md`
* `docs/sample-dataset.md`
* `docs/exploratory-report.md`
* `docs/baseline-targets.md`
* `docs/next-actions.md`

Update `docs/next-actions.md` whenever completing or changing the immediate work queue.

## Data Rules

* Limit project data to Italy.
* Store raw source records unchanged before normalization.
* Normalize timestamps to UTC.
* Preserve source identifiers, source URIs, and uncertainty fields.
* Preserve modality provenance for seismic, VLF, and astronomical features.
* Keep source connectors separate from normalization, feature generation, modeling, and evaluation.

## Source Validation

Mark a source usable only after a reproducible sample pull works.

For INGV, prefer the public FDSN event service documented in `docs/source-inventory.md`. Some query variants may fail; record the exact working request shape before relying on it.

For VLF, Cumiana live JPG spectrograms are the first confirmed capture candidate. Treat them as image-derived features unless raw waveform or numeric trace exports are found.

For astronomical and geomagnetic data, NOAA SWPC live JSON and USNO moon phase JSON are confirmed candidates. Confirm archival access before using them for historical backtests.

## Modeling Rules

Start with naive and historical-rate seismic baselines, then compare against multimodal models. Do not claim value from VLF or astronomical features without ablation tests on held-out data.

Use time-based validation first. Training data must occur before validation data.

## Python Environment

Ubuntu may block user-level `pip` installs through PEP 668. Use a project virtual environment for non-apt Python packages.

Recommended setup:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install numba pyvista h5py zarr
```

Install `netCDF4` in the venv too if running the full offline test suite there:

```sh
pip install netCDF4
```

Run project commands with the venv activated when using optional simulation or visualization dependencies.

## Simulation Environment

The current system has no GPU. Keep sandpile simulation and related smoke tests CPU-only, using Numba CPU kernels. Do not add CUDA, CuPy, or GPU-only ML dependencies unless the runtime target changes.
