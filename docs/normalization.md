# Source Data Normalization

This document defines normalization requirements, field mappings, and command examples for transforming raw ingest files (seismic and space-weather/astronomical data) into clean, structured tables.

---

## 1. INGV Seismic Event Normalization

Converts raw text/export files under `data/raw/ingv/` into structured regional CSV tables. Keep raw files unchanged.

### Execution Commands
To normalize all Italy events:
```sh
PYTHONPATH=src python3 -m elfquake.cli normalize-ingv-events --raw data/raw/ingv/events_italy_2026-06-22.txt --out data/derived/ingv/events_italy_2026-06-22.normalized.csv
```

To normalize with a bounding box filter for `central_italy` only:
```sh
PYTHONPATH=src python3 -m elfquake.cli normalize-ingv-events --raw data/raw/ingv/events_italy_2026-06-22.txt --out data/derived/ingv/events_central_italy_2026-06-22.normalized.csv --only-region central_italy
```

### Seismic Field Mapping
| Raw Ingest Field | Normalized Output Field | Normalization Rule |
| --- | --- | --- |
| `EventID` | `event_id` | Keep string format. |
| `Time` | `event_time_utc` | Treat input as UTC and append trailing `Z` if missing. |
| `Latitude` | `latitude` | Decimal degrees. |
| `Longitude` | `longitude` | Decimal degrees. |
| `Depth/Km` | `depth_km` | Depth in kilometers. |
| `MagType` | `magnitude_type` | Preserve raw string value. |
| `Magnitude` | `magnitude` | Keep numeric value. |
| `EventLocationName` | `event_location_name` | Preserve raw string value. |
| `EventType` | `event_type` | Preserve raw string value. |

### Derived Metadata Columns
*   `source`: Set to `ingv_fdsn_event_text`
*   `italy_region`: Set to `central_italy` or `unknown` (based on bounding box rule below)
*   `raw_file`: Local path of the source raw file
*   `ingested_at_utc`: Timestamp of the ingest pull
*   `raw_uri`: Ingestion URL request string

### Central Italy Geographic Boundary Rule
For the Italy smoke dataset, assign `italy_region = central_italy` if:
*   Latitude is in `[41.5, 43.5]`
*   Longitude is in `[12.0, 14.5]`
Otherwise, set to `unknown`.

---

## 2. Space Weather and Astronomical Normalization

Archive normalizers live in `src/elfquake/normalize/space_weather.py`. They extract clean parameters from raw astronomical, geomagnetic, and solar flux records.

### Space Weather Normalization Rules
*   **GFZ Kp/Ap Text**: Parses planetary records into `date,slot,kp,ap,source_file`.
*   **Kyoto Dst Text**: Parses monthly hourly records into `date,hour,dst_nt,source_file`.
*   **F10.7 Solar Flux**: Parses daily flux table lists into `date,f107,source_file`.
*   **GOES XRS NetCDF**: Requires `netCDF4` library. Extracts `time_utc,variable,value,units,source_file`.
    *   *Usage*: Use `--start` and `--end` to slice sub-windows from large yearly NetCDF files, and `--max-rows` to bound checks.

### execution Commands
To normalize Space Weather Canada daily F10.7:
```sh
PYTHONPATH=src python3 -m elfquake.cli normalize-f107-daily --raw data/raw/astronomy/fluxtable.txt --out data/derived/multimodal/spaceweather_canada_f107.normalized.csv
```

To normalize GFZ Kp/Ap archives:
```sh
PYTHONPATH=src python3 -m elfquake.cli normalize-gfz-kp-ap --raw data/raw/astronomy/Kp_ap_since_1932.txt --out data/derived/multimodal/gfz_kp_ap.normalized.csv
```

---

## 3. Dependency Requirements
System packages are required to compile NetCDF tools for python environments:
*   **OS Level (Ubuntu/Debian)**: `sudo apt install python3-netcdf4`
*   **Python Virtual Environment**: `pip install -r requirements-optional.txt` (installs `netCDF4` and standard dependencies).
