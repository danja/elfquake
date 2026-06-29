# Data Acquisition Code

Initial acquisition code is stdlib-only Python under `src/elfquake`.

## Commands

Run from the repository root with `PYTHONPATH=src`.

Fetch recent INGV Italy events:

```sh
PYTHONPATH=src python -m elfquake.cli fetch-ingv-events --start 2026-06-22T00:00:00Z --end 2026-06-29T23:59:59Z
```

Fetch Cumiana VLF live images:

```sh
PYTHONPATH=src python -m elfquake.cli fetch-vlf-cumiana
```

Run a polite Cumiana VLF cadence check:

```sh
PYTHONPATH=src python -m elfquake.cli capture-vlf-cumiana-loop --only last_E_VLF --cycles 2 --interval-seconds 1800
```

Fetch astronomical and geomagnetic JSON:

```sh
PYTHONPATH=src python -m elfquake.cli fetch-astronomy --date 2026-06-29
```

Run local scaffold tests:

```sh
PYTHONPATH=src python -m unittest discover -s tests
```

Run live endpoint tests separately:

```sh
ELFQUAKE_LIVE_TESTS=1 PYTHONPATH=src python -m unittest discover -s tests_live
```

Live tests make network requests but avoid large archive downloads.

Build the first multimodal smoke row:

```sh
PYTHONPATH=src python -m elfquake.cli build-multimodal-smoke --events data/derived/ingv/events_central_italy_2026-06-15_2026-06-29_2026-06-29T10-24-38Z.normalized.csv --vlf-metadata data/raw/vlf/cumiana/captures/2026-06-29/last_E_VLF_2026-06-29T09-45-00Z.jpg.metadata.json --astronomy-metadata data/raw/astronomy/captures/2026-06-29/usno_moon_phases_2026-06-29T09-56-53Z.json.metadata.json --astronomy-metadata data/raw/astronomy/captures/2026-06-29/noaa_solar_cycle_f107_2026-06-29T10-10-17Z.json.metadata.json --region-id central_italy --window-start 2026-06-22T00:00:00Z --window-end 2026-06-29T10:15:00Z --target-end 2026-07-06T10:15:00Z --out data/derived/multimodal/central_italy_2026-06-22_2026-06-29T1015.multimodal_smoke.csv
```

Build standalone modality feature rows:

```sh
PYTHONPATH=src python -m elfquake.cli build-vlf-features --metadata data/raw/vlf/cumiana/captures/2026-06-29/last_E_VLF_2026-06-29T09-45-00Z.jpg.metadata.json --window-start 2026-06-29T09:00:00Z --window-end 2026-06-29T10:00:00Z --out data/derived/multimodal/cumiana_vlf_2026-06-29T0900_1000.features.csv
PYTHONPATH=src python -m elfquake.cli build-astronomy-features --metadata data/raw/astronomy/captures/2026-06-29/usno_moon_phases_2026-06-29T09-56-53Z.json.metadata.json --metadata data/raw/astronomy/captures/2026-06-29/noaa_solar_cycle_f107_2026-06-29T10-10-17Z.json.metadata.json --window-start 2026-06-29T09:00:00Z --window-end 2026-06-29T10:15:00Z --out data/derived/multimodal/astronomy_2026-06-29T0900_1015.features.csv
```

## Storage

Each capture writes the raw payload and a sibling `.metadata.json` file containing source URL, status, capture time, and response headers.

See [Acquisition Smoke Run](acquisition-smoke-run.md) for the first successful live captures.

Network and HTTP failures return exit code `2` with a concise error message.

## Boundaries

Connectors only acquire and store raw data. Normalization, feature extraction, modeling, and evaluation should remain separate modules.
