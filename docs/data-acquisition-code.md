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

Fetch astronomical and geomagnetic JSON:

```sh
PYTHONPATH=src python -m elfquake.cli fetch-astronomy --date 2026-06-29
```

Run local scaffold tests:

```sh
PYTHONPATH=src python -m unittest discover -s tests
```

## Storage

Each capture writes the raw payload and a sibling `.metadata.json` file containing source URL, status, capture time, and response headers.

See [Acquisition Smoke Run](acquisition-smoke-run.md) for the first successful live captures.

Network and HTTP failures return exit code `2` with a concise error message.

## Boundaries

Connectors only acquire and store raw data. Normalization, feature extraction, modeling, and evaluation should remain separate modules.
