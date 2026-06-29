# Systemd Service

Service unit:

`deploy/systemd/elfquake.service`

Optional environment file:

`deploy/systemd/elfquake.env`

Purpose:

* run the Cumiana VLF capture loop as a long-running service
* capture `last_E_VLF` every 30 minutes
* write only under `data/`

Install manually:

```sh
sudo cp deploy/systemd/elfquake.service /etc/systemd/system/elfquake.service
sudo cp deploy/systemd/elfquake.env /etc/default/elfquake
sudo systemctl daemon-reload
sudo systemctl enable --now elfquake.service
```

Check status:

```sh
systemctl status elfquake.service
journalctl -u elfquake.service -f
```

Adjust `User`, `Group`, `WorkingDirectory`, and `PYTHONPATH` if the repo is moved.
Adjust `/etc/default/elfquake` to change `ELFQUAKE_VLF_INTERVAL_SECONDS`.

The service uses:

```sh
PYTHONPATH=src python3 -m elfquake.cli capture-vlf-cumiana-loop --only last_E_VLF --cycles 0 --interval-seconds 1800
```
