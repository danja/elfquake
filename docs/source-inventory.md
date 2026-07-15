# Source Inventory and Onboarding

This document tracks Italy sources and the Japan parallel experiment, details the criteria to consider a source usable, and specifies the onboarding checklist for new sources. The production study remains Italy-scoped; Japan is an isolated comparative dataset until its passive-radio coverage is verified.

## 1. Master Source Inventory

Italy remains the production scope. Japan sources are maintained in a separate namespace for comparative experiments. A source is marked **Usable** only after a raw sample pull is verified as reproducible.

| Source | Use | Access / URL | Format | Status |
| --- | --- | --- | --- | --- |
| **INGV FDSN event** | Italian earthquake catalog | `https://webservices.ingv.it/fdsnws/event/1/` | text | **Usable**: tested for recent Italy exports; see connector notes. |
| **INGV FDSN station**| Station metadata | `https://webservices.ingv.it/fdsnws/station/1/` | StationXML, text | *Candidate* |
| **INGV FDSN dataselect**| Waveform signals | `https://webservices.ingv.it/fdsnws/dataselect/1/` | MiniSEED | *Candidate* |
| **vlf.it Cumiana live**| VLF radio context | `http://www.vlf.it/cumiana/livedata.html` | JPG spectrograms | **Usable**: live image endpoints confirmed; raw waveforms pending. |
| **Abelian Cumiana VLF**| VLF audio & archives | `http://abelian.org/vlf/` | Ogg live stream, retrieve form | *Candidate*: live/archive endpoints respond, but empty/zero-byte pulls. |
| **NOAA SWPC** | Solar & geomagnetic indices | `https://services.swpc.noaa.gov/json/` | JSON | **Usable**: monthly solar cycle; rolling indexes (Kp, Dst, GOES X-ray). |
| **NOAA/NCEI GOES SEM**| GOES XRS science archives| `https://www.ncei.noaa.gov/instruments/solar-space-observing/particle-detectors/sem/goes/access/` | NetCDF | **Usable**: historical GOES-15 NetCDF archive (2010–2020) confirmed. |
| **GFZ Kp** | Planetary Kp/ap archive | `https://kp.gfz.de/app/files/Kp_ap_since_1932.txt` | text | **Usable**: historical Kp/ap data since 1932 confirmed. |
| **Kyoto WDC Dst** | Dst hourly index | `https://wdc.kugi.kyoto-u.ac.jp/dstdir/` | text, HTML | **Usable**: final (to 2020) and provisional (to 2026) pages confirmed. |
| **Space Weather Canada**| Daily F10.7 solar flux | `https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt` | text | **Usable**: daily archive since 2004 confirmed. |
| **USNO** | Lunar phase events | `https://aa.usno.navy.mil/api/` | JSON | **Usable**: Moon phase events API confirmed. |
| **USGS FDSN event** | Japan regional earthquake catalog | `https://earthquake.usgs.gov/fdsnws/event/1/` | GeoJSON | **Candidate**: connector and bounded sample path implemented; verify a live sample before modelling. |
| **Japan ISEE Moshiri/Kagoshima** | Passive natural-radio context | `https://stdb2.isee.nagoya-u.ac.jp/vlf/` | broadband ELF/VLF waveform/spectrogram | **Candidate verified**: stations, antenna type, sampling and coverage documented; raw digital access requires request. |
| **Japan/WALDO broadband VLF** | Historical passive-radio fallback | `https://waldo.world/` | MATLAB v4-like broadband waveform | **Deferred**: old, browser/account-dependent, and unsuitable as the primary current Japan source. See [waldo.md](waldo.md). |

---

## 2. Italy Region Bounding Filter

Use an explicit geographic filter for Italy on all spatial sources. We start with a bounding box and will replace it with precise regional polygons when finer spatial labels are required.

*   **Latitude Bounds**: `35` to `48`
*   **Longitude Bounds**: `6` to `19`

## 2a. Japan Region Bounding Filter

The initial Japan experiment uses `30` to `46` degrees latitude and `129` to
`146` degrees longitude. This is an acquisition box, not a claim that every
event in the box belongs to one tectonic regime.

---

## 3. Source Checklist Criteria

When evaluating or onboarding a new data source, document and verify the following criteria:

1.  **Access Method**: API endpoint pattern, query parameters, and update frequency/cadence.
2.  **Licensing**: Reuse or commercial distribution constraints (e.g. Kyoto Dst non-commercial notice, GFZ CC BY 4.0).
3.  **Schema / Format**: Raw response structure (JSON, NetCDF, CSV, text).
4.  **Time Domain**: Precision, resolution, timezone (always normalize to UTC), and alignment keys.
5.  **Spatial Scope**: Bounding box alignment or station coordinates.
6.  **Quality / Completeness**: Document known gaps, service outages, or missing periods.

---

## 4. Ingestion and Normalization Guidelines

*   **Immutable Raw Data**: Store raw data payloads exactly as received from the source. Do not edit raw records.
*   **Decoupled Connectors**: Keep source connectors separate from normalization and feature extraction.
*   **Explicit Provenance**: Include raw source files, ingestion timestamps, and request URLs in the metadata of all normalized tables.

---

## References
*   See [VLF Feasibility](vlf-feasibility.md) for details on Cumiana and Abelian VLF data.
*   See [Astronomical Feasibility](astronomical-feasibility.md) for details on solar, lunar, and geomagnetic archives.
