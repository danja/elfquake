# Astronomical and Geomagnetic Feasibility

This document consolidates feasibility findings, useful features, endpoints, and archive details for lunar phase, solar radio flux, and geomagnetic indices.

## 1. Feasibility Findings and Decisions

*   **Lunar Phase Data**: Fully confirmed. We use USNO APIs to generate lunar cycle context relative to prediction windows.
*   **Solar and Geomagnetic Indices**: Fully confirmed. While live NOAA SWPC JSON files are rolling, historical archives exist for solar radio flux (F10.7) and Kp planetary indices.
*   **Decision**: Use NOAA solar-cycle monthly historical JSON for early coarse features. Do not use NOAA SWPC rolling Kp, Dst, GOES X-ray, or daily F10.7 feeds for historical backtests when historical archive endpoints are available.

---

## 2. Ingest Source Inventory

| Modality / Source | URL / Endpoint | Coverage / Resolution | Backtest / Archive Status |
| --- | --- | --- | --- |
| **USNO Moon Phase** | `https://aa.usno.navy.mil/api/moon/phases/date` | Specific phase event dates (new, first quarter, full, last quarter) | **Usable**: Moon phase endpoint is stable and queryable. |
| **NOAA Solar Cycle (Monthly)** | `https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json` | Historical monthly sunspots and F10.7 (2004–2026) | **Usable**: Coarse historical context. |
| **GFZ Kp/ap Index** | `https://kp.gfz.de/app/files/Kp_ap_since_1932.txt` | 3-hour planetary Kp/ap slots since 1932 | **Usable**: CC BY 4.0 licensed historical archive. |
| **Kyoto WDC Dst (Final)** | `https://wdc.kugi.kyoto-u.ac.jp/dst_final/` | Monthly hourly Dst indices (1957–2020) | **Usable** (non-commercial licence restriction). |
| **Kyoto WDC Dst (Provisional)**| `https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/` | Monthly hourly Dst indices (2021–2026) | **Usable** (non-commercial licence restriction). |
| **NOAA/NCEI GOES XRS** | `https://www.ncei.noaa.gov/instruments/solar-space-observing/particle-detectors/sem/goes/access/science/xrs/` | Science-quality GOES 8–15 XRS NetCDF files (e.g., 2010–2020) | **Usable**: High-resolution X-ray flux profiles. |
| **Space Weather Canada F10.7**| `https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt` | Daily F10.7 solar radio flux since 2004 | **Usable**: Detailed daily solar activity indicator. |
| *NOAA SWPC Kp (1m)* | `https://services.swpc.noaa.gov/json/planetary_k_index_1m.json` | 1-minute rolling planetary index | *Not archival* (live feed only). |
| *NOAA GOES X-ray (1m)* | `https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json` | 7-day rolling 1-minute averages | *Not archival* (live feed only). |

---

## 3. Window Feature Alignment

For each prediction/feature window (e.g. 7-day or hourly window), we extract the following features:

*   **Lunar Context**: Distance in days/hours to the nearest full moon and new moon.
*   **Geomagnetic Activity**: Mean, maximum, variance, and missing count of the Kp and ap indices.
*   **Solar Activity**: Daily or monthly F10.7 solar radio flux (observed and adjusted) and monthly sunspot number (SSN).
*   **X-Ray Flux**: GOES XRS science flux averages, peak magnitudes, and energy band distributions.

---

## 4. Normalization and Code Interfaces

To fetch daily astronomy/space-weather sources (e.g., USNO moon phases and NOAA solar-cycle indices):
```sh
PYTHONPATH=src python3 -m elfquake.cli fetch-astronomy --date 2026-06-29
```

To fetch GFZ Kp/ap historical text archive:
```sh
PYTHONPATH=src python3 -m elfquake.cli fetch-gfz-kp-ap
```

To fetch Space Weather Canada daily F10.7:
```sh
PYTHONPATH=src python3 -m elfquake.cli fetch-f107-daily
```

