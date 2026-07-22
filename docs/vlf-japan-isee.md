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
* Nonempty CDF sample: verified locally from Moshiri, `2025-01-01 00:00 UTC` to approximately `01:00 UTC`.
* Decoder: `normalize-vlf-japan-cdf`, using optional `cdflib`.

The decoder preserves the raw CDF, identifies an epoch variable, writes scalar time-series variables to CSV, and records global/variable attributes and non-scalar spectrum dimensions in JSON. It does not flatten spectra or assume that station channels, units, or frequency axes are interchangeable.

The first sample has 8,646 records at 0.4096-second resolution. `ch1` and `ch2` are 8,646 by 1,024 spectrogram arrays with frequency support from 10 Hz to 19,989 Hz and units `V^2/Hz`. The CDF identifies the station as Moshiri (`44.37 N, 142.27 E`) and describes the crossed-loop antenna channels as orthogonal magnetic-field components. The raw file checksum is recorded in its `.capture.json` sidecar.

## Workflow

1. Obtain one exact CDF file URL and record the station, date range, sampling, antenna orientation, and scientific-use terms. The first verified sample is listed in `data/raw/vlf/japan/manifest.csv`.
2. Store the unchanged file under `data/raw/vlf/japan/` with `URL=<exact-file-url> ./scripts/fetch-japan-vlf-cdf.sh`.
3. Install `cdflib`, then run `INPUT=data/raw/vlf/japan/<file>.cdf ./scripts/normalize-japan-vlf-cdf.sh`.
4. Inspect the metadata and compare the decoded time axis with the station documentation before feature generation.
5. Keep Japan evaluation separate from Italy model scores unless a cross-region experiment is explicitly declared.
