---
name: elfquake-source-ingest
description: Use when running one-off ELFQuake source fetch, capture, normalization, or feature-build CLI commands for INGV, VLF, astronomy, geomagnetic, or space-weather data.
---

# ELFQuake Source Ingest

Run from the repository root. Use `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli ...` unless an existing shell wrapper is more appropriate.

## Acquisition

Use bounded, reproducible pulls and keep raw files unchanged:

```sh
./backfill-ingv-history.sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli plan-ingv-backfill
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli fetch-ingv-events
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli fetch-vlf-cumiana
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli record-vlf-abelian-cumiana
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli fetch-astronomy
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli fetch-gfz-kp-ap
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli fetch-f107-daily
```

Add explicit `--start`, `--end`, `--year`, `--date`, or source-selection options when a workflow needs a precise sample. Record exact working request shapes in docs when source behavior is uncertain.

## Normalization And Features

Normalize before joining or modeling:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli normalize-ingv-events
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli combine-normalized-events
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli normalize-gfz-kp-ap
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli normalize-f107-daily
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli extract-vlf-image-features
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m elfquake.cli build-astronomy-features
```

Keep connectors, normalization, feature generation, model input materialization, and evaluation separate. For network failures in this environment, retry only with user-approved escalation.
