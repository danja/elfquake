# Astronomical Feasibility

Early feasibility check for lunar, solar, and geomagnetic context.

## Confirmed Candidates

| Source | Endpoint | Cadence | Status |
| --- | --- | --- | --- |
| NOAA SWPC Kp | `https://services.swpc.noaa.gov/json/planetary_k_index_1m.json` | 1 minute | live JSON confirmed |
| NOAA GOES X-ray | `https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json` | 1 minute | live JSON confirmed |
| NOAA solar cycle | `https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json` | monthly | historical JSON confirmed |
| USNO moon phases | `https://aa.usno.navy.mil/api/moon/phases/date?date=YYYY-MM-DD&nump=N` | phase events | JSON confirmed |

## Useful Fields

* Kp: `time_tag`, `kp_index`, `estimated_kp`, `kp`
* GOES X-ray: `time_tag`, `satellite`, `flux`, `observed_flux`, `energy`
* Solar cycle: `time-tag`, `ssn`, `f10.7`
* Moon phase: `year`, `month`, `day`, `time`, `phase`

## Window Alignment

For each seismic/VLF window, derive:

* Kp min, max, mean, and missing count
* GOES X-ray flux summaries by energy band
* monthly sunspot and F10.7 values
* nearest lunar phase before and after the window
* days from full moon and new moon

## Gaps

NOAA live endpoints are useful for current capture, but most SWPC Kp, Dst, GOES X-ray, and daily F10.7 feeds checked so far are rolling files. Do not rely on them for backtests until archival coverage is confirmed.

See [NOAA Archive Feasibility](noaa-archive-feasibility.md).

Current coarse feature stub: [Feature Extraction](feature-extraction.md).
