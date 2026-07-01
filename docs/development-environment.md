# Development Environment

Use a project virtual environment for non-apt Python packages. Ubuntu may block user-level `pip` installs through PEP 668.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install numba pyvista h5py zarr netCDF4
```

Run optional simulation, visualization, and full offline test commands with the venv activated.

The current target system has no GPU. Keep sandpile simulation and smoke tests CPU-only; do not add CUDA, CuPy, or GPU-only ML dependencies unless the runtime target changes.
