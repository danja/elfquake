# ISEE Japan VLF

Nagoya University ISEE has supplied three official plot/viewer entry points and an ERG Science Center archive of digital VLF spectra in CDF format:

* [VLF viewer](https://stdb2.isee.nagoya-u.ac.jp/vlf/index2.html)
* [Dynamic spectrum viewer](https://stdb2.isee.nagoya-u.ac.jp/vlf/spectrum/)
* [Ground-observation data policy and catalog](https://ergsc.isee.nagoya-u.ac.jp/data_info/ground.shtml.ja#jump14)
* [VLF CDF archive](https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/ground/vlf/)

The archive lists passive ground VLF/ELF stations including Athabasca, Gakona, Husafell, Istok, Kapuskasing, Maimaga, Moshiri, Nain, and Oulujarvi. The source is scientifically restricted and is not part of the Italy production dataset. The project contact has confirmed that this work is permitted for scientific use; the remaining gate is a reproducible file pull with station metadata and units.

## Current Status

* Plot and archive pages: verified.
* Machine-readable CDF archive: verified.
* Scientific-use permission: confirmed by the project contact; retain the archive caveats with any derived result.
* Nonempty CDF samples: two verified locally from Moshiri, January and June 2025.
* Japan seismic alignment: 319 normalized USGS events and 26 mature weekly windows; both January and June samples now have one overlapping mature window.
* Decoder: `normalize-vlf-japan-cdf`, using optional `cdflib`.

The decoder preserves the raw CDF, identifies an epoch variable, writes scalar time-series variables to CSV, and records global/variable attributes and non-scalar spectrum dimensions in JSON. It does not flatten spectra or assume that station channels, units, or frequency axes are interchangeable.

The first sample has 8,646 records at 0.4096-second resolution. `ch1` and `ch2` are 8,646 by 1,024 spectrogram arrays with frequency support from 10 Hz to 19,989 Hz and units `V^2/Hz`. The CDF identifies the station as Moshiri (`44.37 N, 142.27 E`) and describes the crossed-loop antenna channels as orthogonal magnetic-field components. The raw file checksum is recorded in its `.capture.json` sidecar.

`./scripts/extract-japan-vlf-cdf-features.sh` converts the native arrays into one row per timestamp with eight logarithmic frequency bands per channel, active-bin fractions, valid-bin fractions, and `research_use_only=1`. The output is a compact feature representation for experiments; the raw CDF remains the authoritative record.

`./scripts/build-japan-vlf-cdf-window-features.sh` aggregates one or more CDF feature files to the project window contract using mean, standard deviation, maximum, row count, and coverage duration. It is the boundary between native Japan data and model-ready windows. `./scripts/build-japan-vlf-cdf-dataset.sh` discovers all processed CDF feature files and creates one combined row per Japan seismic window.

`./scripts/process-japan-vlf-manifest.sh` is the repeatable ingestion workflow. It processes every CDF row in the manifest, skips already captured raw files, and optionally builds window features when `WINDOWS` is supplied. It is safe to rerun and all Japan outputs remain research-only.

For unattended collection, `deploy/systemd/elfquake-japan-vlf.service` and `.timer` run `scripts/refresh-japan-vlf.sh` every six hours. The refresh discovers only the newest unrecorded CDF from the previous archive month and defaults to one file per run, limiting storage growth and network load. Downloads use a temporary file and atomic rename, so interrupted transfers are retried rather than treated as valid CDFs. This service is independent of the Italy VLF service.

The alignment run produces 3,589 native VLF rows in one January weekly window and 3,589 in one June weekly window. This confirms temporal plumbing for both samples, but is still far too little coverage for model training or scientific association claims.

## Workflow

1. Obtain one exact CDF file URL and record the station, date range, sampling, antenna orientation, and scientific-use terms. The verified samples are listed in `data/raw/vlf/japan/manifest.csv`.
2. Store the unchanged file under `data/raw/vlf/japan/` with `URL=<exact-file-url> ./scripts/fetch-japan-vlf-cdf.sh`.
3. Install `cdflib`, then run `INPUT=data/raw/vlf/japan/<file>.cdf ./scripts/normalize-japan-vlf-cdf.sh`.
4. Run `INPUT=data/raw/vlf/japan/<file>.cdf ./scripts/extract-japan-vlf-cdf-features.sh` after checking the metadata.
5. Inspect the feature metadata and compare the decoded time axis with the station documentation.
6. For repeatable processing, run `WINDOWS=data/derived/japan/japan.seismic_training_windows.csv ./scripts/process-japan-vlf-manifest.sh`.
7. Use all Japan raw data and derived features only for scientific research; retain the archive caveats and contact requirement.
8. Keep Japan evaluation separate from Italy model scores unless a cross-region experiment is explicitly declared.

## Systemd Installation

```sh
sudo cp deploy/systemd/elfquake-japan-vlf.service /etc/systemd/system/
sudo cp deploy/systemd/elfquake-japan-vlf.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now elfquake-japan-vlf.timer
```

Inspect with `systemctl list-timers elfquake-japan-vlf.timer` and `journalctl -u elfquake-japan-vlf.service`. A new hourly CDF is typically around 6--11 MB and may take several minutes to download. For a manual run without waiting in the terminal, use `sudo systemctl start --no-block elfquake-japan-vlf.service`. A successful run ends with `Deactivated successfully` and `Finished`. Japan raw and derived data are for scientific research only.

The unit disables Numba JIT because the service imports the shared CLI, including simulation commands, but never runs simulation. This avoids Numba cache initialization under systemd and does not affect the CDF features.
