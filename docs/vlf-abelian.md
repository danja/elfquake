# Abelian VLF

Candidate source for live and archived natural-radio samples.

## Endpoints

| Endpoint | Use | Status |
| --- | --- | --- |
| `http://abelian.org/vlf/` | live receiver page | Cumiana `vlf15` listed |
| `http://abelian.org/vlf/live-stream.php?stream=vlf15` | Cumiana live Ogg audio | endpoint confirmed; first body was zero bytes |
| `http://abelian.org/vlf/retrieve.php` | archive retrieval form | form confirmed |

## Cumiana Metadata

* station: Cumiana, NW Italy
* stream id: `vlf15`
* coordinates: `44.96`, `7.42`
* operator: Renato Romero / Openlab

## Archive Form

The retrieval form uses GET parameters:

* `ts`: UTC start, `YYYY-MM-DD HH:MM:SS.SSSSSS`
* `len`: seconds, `0.05` to `120.0`
* `vlf15=on`: include Cumiana
* `format`: `sg`, `td`, `vt`, or `wav`

Submitted responses can include generated download links under `/vlf/live/retrieve/`.

## Commands

Record a short live audio chunk. The command fails if the endpoint returns no bytes:

```sh
PYTHONPATH=src python -m elfquake.cli record-vlf-abelian-cumiana --duration-seconds 10 --max-bytes 1048576
```

Fetch an archive request and any nonempty generated download:

```sh
PYTHONPATH=src python -m elfquake.cli fetch-vlf-abelian-cumiana-archive --start 2026-07-05T10:38:11Z --duration-seconds 5 --format wav
```

Write a CSV source-validation probe:

```sh
PYTHONPATH=src python -m elfquake.cli probe-vlf-abelian-cumiana-archive --start 2026-07-05T10:38:11Z --duration-seconds 0.05 --format wav --format vt --out data/derived/vlf/abelian_cumiana_archive_probe_2026-07-05.csv
```

Extract coarse audio/container features:

```sh
PYTHONPATH=src python -m elfquake.cli extract-vlf-audio-features --audio-root data/raw/vlf/abelian/cumiana --out data/derived/vlf/abelian_cumiana_audio.features.csv
```

## Decision

Treat Abelian as a promising raw VLF candidate, not yet a validated dataset. Mark it usable only after reproducible nonempty live or archive pulls cover time ranges useful for INGV alignment.

## Probe Results

`data/derived/vlf/abelian_cumiana_archive_probe_2026-07-05.csv` tested Cumiana `vlf15` at `2026-07-05T10:38:11Z` for `wav` and `vt`. Both responses were HTTP 200, but both reported `no database`, one generated link, declared download size `0`, and `usable_nonempty=0`.

`data/derived/vlf/abelian_cumiana_archive_probe_multi_2026-06-29_2026-07-05.csv` tested five additional timestamps from `2026-06-29` through `2026-07-05` for `wav` and `vt`. All ten rows were HTTP 200, but all reported `no database`, declared download size `0`, and `usable_nonempty=0`.
