# Development Environment

Use a project virtual environment for non-apt Python packages. Ubuntu may block user-level `pip` installs through PEP 668.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-optional.txt
```

Run optional simulation, visualization, and full offline test commands with the venv activated.

CPU PyTorch is optional but required for the neural tabular baseline:

```sh
pip install torch
```

If NumPy has been upgraded to a version incompatible with Numba, repair the venv with:

```sh
pip install --upgrade --force-reinstall --no-cache-dir -r requirements-optional.txt
```

The current target system has no GPU. Keep sandpile simulation and smoke tests CPU-only; do not add CUDA, CuPy, or GPU-only ML dependencies unless the runtime target changes.
