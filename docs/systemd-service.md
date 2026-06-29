# Systemd Service

Service unit:

`deploy/systemd/elfquake.service`

Prospective row timer:

* `deploy/systemd/elfquake-prospective.service`
* `deploy/systemd/elfquake-prospective.timer`

Optional environment file:

`deploy/systemd/elfquake.env`

Purpose:

* run the Cumiana VLF capture loop as a long-running service
* capture `last_E_VLF` every 30 minutes
* optionally update cumulative prospective VLF rows and image-derived VLF features every 30 minutes
* write only under `data/`

Install manually:

```sh
sudo cp deploy/systemd/elfquake.service /etc/systemd/system/elfquake.service
sudo cp deploy/systemd/elfquake-prospective.service /etc/systemd/system/elfquake-prospective.service
sudo cp deploy/systemd/elfquake-prospective.timer /etc/systemd/system/elfquake-prospective.timer
sudo cp deploy/systemd/elfquake.env /etc/default/elfquake
sudo systemctl daemon-reload
sudo systemctl enable --now elfquake.service
sudo systemctl enable --now elfquake-prospective.timer
```

Check status:

```sh
systemctl status elfquake.service
systemctl list-timers elfquake-prospective.timer
journalctl -u elfquake.service -f
journalctl -u elfquake-prospective.service -f
```

Adjust `User`, `Group`, `WorkingDirectory`, and `PYTHONPATH` if the repo is moved.
Adjust `/etc/default/elfquake` to change `ELFQUAKE_VLF_INTERVAL_SECONDS`, lookback horizon, target magnitude, or table paths.

The service uses:

```sh
PYTHONPATH=src python3 -m elfquake.cli capture-vlf-cumiana-loop --only last_E_VLF --cycles 0 --interval-seconds 1800
```

The prospective timer uses:

```sh
PYTHONPATH=src python3 -m elfquake.cli update-prospective-vlf-table ...
```

It also runs:

```sh
PYTHONPATH=src python3 -m elfquake.cli extract-vlf-image-features --image-root data/raw/vlf/cumiana/captures --filename-prefix last_E_VLF ...
PYTHONPATH=src python3 -m elfquake.cli join-vlf-image-features ...
PYTHONPATH=src python3 -m elfquake.cli summarize-prospective-table ...
```
