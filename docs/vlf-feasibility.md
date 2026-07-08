# VLF Radio Source Feasibility

This document consolidates feasibility findings, receiver stations, endpoints, acquisition/capture manifests, and coverage status for Italy-relevant Very Low Frequency (VLF) radio data.

## 1. Core Feasibility Findings

*   **VLF.it Homepage**: Reachable over HTTP, serving as a static library of articles and amateur radio experiments, rather than a documented web service. Open Lab guidelines allow non-commercial use, but commercial rights are reserved by the authors.
*   **Cumiana Station**: Confirmed machine-fetchable live spectrograms (JPEG) and status summaries.
*   **Abelian Natural Radio**: Exposes Cumiana Ogg streaming and a retrieval PHP form. However, early probes returned zero usable bytes, and the archive returns `no database` errors.
*   **Modeling Readiness**: Image-derived parameters are used as proxy features until raw waveforms or numeric trace endpoints are established.

---

## 2. Cumiana VLF Monitoring Station

*   **Location**: Cumiana, Torino, NW Italy (`44.95609` N, `7.42123` E approximate)
*   **Maintainer**: Renato Romero / Openlab
*   **Page**: `http://www.vlf.it/cumiana/livedata.html` (requires `curl --http1.1 --raw` configuration).

### Live Spectrogram Endpoints
Spectrogams are updated roughly every 30 or 60 minutes:

| Endpoint ID | URL | Content | Cadence |
| --- | --- | --- | --- |
| `last_E_VLF` | `http://www.vlf.it/cumiana/last_E-VLF.jpg` | Last 8 hours vertical electric-field spectrogram (Marconi antenna) | 30 mins |
| `last-geomar` | `http://www.vlf.it/cumiana/last-geomar.jpg` | Geophone (1–30 Hz) + electric-field (1–105 Hz) spectrogram | 30 mins |
| `last-marconi-multistrip-slow` | `http://www.vlf.it/cumiana/last-marconi-multistrip-slow.jpg` | Electric-field daily multistrip (110s scroll) | 30 mins |
| `last-geophone-multistrip-slow`| `http://www.vlf.it/cumiana/last-geophone-multistrip-slow.jpg` | Geophone daily multistrip (4.6s scroll) | 60 mins |
| `last-plotted` | `http://www.vlf.it/cumiana/last-plotted.jpg` | Last 30 hours of plotted traces | 30 mins |

---

## 3. Abelian VLF Archive and Live Streams

*   **Live Stream**: `http://abelian.org/vlf/live-stream.php?stream=vlf15` (Cumiana live Ogg stream).
*   **Archive Form**: `http://abelian.org/vlf/retrieve.php`
    *   Parameters: `ts` (UTC start, `YYYY-MM-DD HH:MM:SS`), `len` (seconds, `0.05` to `120.0`), `vlf15=on` (Cumiana filter), `format` (`wav`, `vt`, `sg`, `td`).
*   **Findings**: Archive requests submitted for June/July 2026 timestamps returned HTTP 200 but were empty, reporting `no database` and declaring download size `0`. Do not rely on historical Abelian data until a nonempty download is successfully verified.

---

## 4. Ingestion and Capture Manifest

The Cumiana image capture pipeline is configured via:
*   **Manifest Path**: `data/raw/vlf/cumiana/manifest.csv`
*   **Storage Directory Structure**:
    ```text
    data/raw/vlf/cumiana/captures/YYYY-MM-DD/<endpoint_id>_<last_modified_utc>.jpg
    data/raw/vlf/cumiana/captures/YYYY-MM-DD/<endpoint_id>_<last_modified_utc>.metadata.json
    ```
    *Filenames replace colons with hyphens. The `.metadata.json` stores the response headers (`Date`, `Last-Modified`, `ETag`, `Content-Length`, `Content-Type`).*

### Capture Rules
*   Do not overwrite existing files with the same timestamp.
*   Politely respect the cadence (poll no faster than every 30 or 60 minutes).
*   The supervised loop command rejects repeated runs under 60 seconds.

---

## 5. Feature Extraction and Coverage

Because raw waveforms are not yet available, we extract proxy features by cropping and analyzing the Marconi spectrogram JPEG pixels.

### Extracted Image Features
*   Overall availability and stale/missing flags for each window.
*   Mean and variance within specific cropped time-frequency bands.
*   **Hot-color ratio** and vertical streak count (indicator of impulse noise or sferics).
*   Simple anomaly scores relative to the running baseline of the past week.

### Table Coverage Status
*   **Historical June Labeled Windows**: Have `quality_missing_vlf=1` since active image captures only began after those windows.
*   **Prospective Labeled Windows**: Rebuilt image-window tables (e.g. `central_italy.prospective_vlf_image_windows.labeled.csv`) contain 247 consecutive rows starting from `2026-06-29`, with `0` missing VLF image features.
*   We use the `update-prospective-vlf-table` script to periodically append new rows on a systemd timer.

---

## 6. Acquisition and Verification Commands

Record short live audio stream from Abelian:
```sh
PYTHONPATH=src python3 -m elfquake.cli record-vlf-abelian-cumiana --duration-seconds 10 --max-bytes 1048576
```

Request and retrieve an Abelian archive WAV segment:
```sh
PYTHONPATH=src python3 -m elfquake.cli fetch-vlf-abelian-cumiana-archive --start 2026-07-05T10:38:11Z --duration-seconds 5 --format wav
```

Audit the Abelian archive database for availability:
```sh
PYTHONPATH=src python3 -m elfquake.cli probe-vlf-abelian-cumiana-archive --start 2026-07-05T10:38:11Z --duration-seconds 0.05 --format wav --out data/derived/vlf/abelian_cumiana_archive_probe.csv
```

Extract visual features from a folder of captures:
```sh
PYTHONPATH=src python3 -m elfquake.cli extract-vlf-image-features --image-root data/raw/vlf/cumiana/captures --out data/derived/multimodal/cumiana_last_E_VLF.image_features.csv
```

