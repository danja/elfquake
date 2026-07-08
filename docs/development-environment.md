# Development and Deployment Environment

This document outlines the local setup, virtual environment, systemd production service deployment, and the pre-deployment checklist.

## 1. Local Python Environment

Use a project virtual environment for Python packages to avoid conflicts (and Ubuntu PEP 668 restrictions).

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-optional.txt
```

Run optional simulation, visualization, and full offline test commands with the venv activated.

CPU PyTorch is optional but required for neural tabular/Transformer baselines:

```sh
pip install torch
```

### NumPy and Numba Compatibility
If NumPy has been upgraded to a version incompatible with Numba, repair the venv with:

```sh
pip install --upgrade --force-reinstall --no-cache-dir -r requirements-optional.txt
```

*Note: Numba `0.65.1` requires `numpy<2.5`. Do not let `pip` upgrade NumPy to `2.5.x`.*

### Hardware Target
The current target system has no GPU. Keep sandpile simulation and smoke tests CPU-only; do not add CUDA, CuPy, or GPU-only ML dependencies unless the runtime target changes.

---

## 2. Production Systemd Services

The systemd configuration is defined in:
*   `deploy/systemd/elfquake.service` - Main Cumiana VLF capture loop
*   `deploy/systemd/elfquake-prospective.service` - Prospective row updater
*   `deploy/systemd/elfquake-prospective.timer` - Trigger timer for update-prospective
*   `deploy/systemd/elfquake.env` - Configuration environment file

### Manual Installation
```sh
sudo cp deploy/systemd/elfquake.service /etc/systemd/system/elfquake.service
sudo cp deploy/systemd/elfquake-prospective.service /etc/systemd/system/elfquake-prospective.service
sudo cp deploy/systemd/elfquake-prospective.timer /etc/systemd/system/elfquake-prospective.timer
sudo cp deploy/systemd/elfquake.env /etc/default/elfquake
sudo systemctl daemon-reload
sudo systemctl enable --now elfquake.service
sudo systemctl enable --now elfquake-prospective.timer
```

### Supervision and Logs
```sh
systemctl status elfquake.service
systemctl list-timers elfquake-prospective.timer
journalctl -u elfquake.service -f
journalctl -u elfquake-prospective.service -f
```

*Note: Adjust `User`, `Group`, `WorkingDirectory`, and `PYTHONPATH` in service files if the repository location changes. Adjust `/etc/default/elfquake` to configure timers, lookback horizons, or target thresholds.*

---

## 3. Pre-Real-Data Deployment Checklist

Verify the following before starting production services or live backfills:

*   **Service Installation**: Install and successfully smoke-test `deploy/systemd/elfquake.service`.
*   **System Time**: Confirm the target host clock uses UTC and has reliable NTP enabled.
*   **Manual Fetch Test**: Run one manual test fetch: `PYTHONPATH=src python3 -m elfquake.cli fetch-vlf-cumiana --only last_E_VLF`
*   **Live Tests**: Run live endpoint tests with `ELFQUAKE_LIVE_TESTS=1 PYTHONPATH=src python3 -m unittest discover -s tests_live`.
*   **Ingest Plan**: Review the generated INGV pull plan (e.g., `data/derived/backfill/ingv_italy_2026-06.plan.csv`).
*   **NetCDF Support**: Verify `netCDF4` is importable in the target environment (required for GOES X-ray files).
*   **Disk Policy**: Confirm the storage and retention policy for raw VLF images under `data/raw/vlf/`.

