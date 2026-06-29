# NOAA Archive Feasibility

Official NOAA SWPC/NCEI endpoints and primary external archives checked on `2026-06-29`.

## Findings

| Source | Endpoint | Coverage observed | Backtest status |
| --- | --- | --- | --- |
| Solar cycle monthly | `https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json` | Historical monthly records | Usable for coarse historical context |
| Solar cycle sunspots | `https://services.swpc.noaa.gov/json/solar-cycle/sunspots.json` | Historical monthly records | Candidate |
| Solar cycle F10.7 | `https://services.swpc.noaa.gov/json/solar-cycle/f10-7cm-flux.json` | Historical monthly records, `2004-10` to `2026-05` observed in smoke capture | Usable for coarse historical context |
| GFZ Kp/Ap | `https://kp.gfz.de/app/files/Kp_ap_since_1932.txt` | 3-hour Kp/ap since `1932-01-01`; CC BY 4.0 header observed | Usable candidate |
| Kyoto Dst final | `https://wdc.kugi.kyoto-u.ac.jp/dst_final/` | Monthly final Dst pages from `1957` to `2020`; non-commercial notice observed | Usable with license constraint |
| Kyoto Dst provisional | `https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/` | Monthly provisional Dst pages from `2021-01` to `2026-04`; non-commercial notice observed | Usable with license constraint |
| NOAA/NCEI GOES XRS science | `https://www.ncei.noaa.gov/instruments/solar-space-observing/particle-detectors/sem/goes/access/science/xrs/` | GOES 8-15 science-quality XRS tree; GOES-15 yearly NetCDF `2010` to `2020` observed | Usable candidate |
| NOAA/NCEI GOES SEM full/avg | `https://www.ncei.noaa.gov/instruments/solar-space-observing/particle-detectors/sem/goes/access/` | Year/month archive trees observed from `1974` to `2020` | Candidate |
| Space Weather Canada daily F10.7 | `https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt` | Daily flux table from `2004-10-28` to present | Usable candidate |
| Planetary Kp 1-minute | `https://services.swpc.noaa.gov/json/planetary_k_index_1m.json` | Rolling live file | Not archival |
| Planetary Kp 3-hour | `https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json` | Rolling recent file | Not archival |
| Kyoto Dst | `https://services.swpc.noaa.gov/products/kyoto-dst.json` | Rolling recent file | Not archival |
| GOES X-ray | `https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json` | Rolling 7-day file | Not archival |
| F10.7 daily | `https://services.swpc.noaa.gov/products/10cm-flux-30-day.json` | Rolling 30-day file | Not archival |

## Decision

Use NOAA solar-cycle historical JSON for early coarse astronomical features.

Captured sample:

`data/raw/astronomy/captures/2026-06-29/noaa_solar_cycle_f107_2026-06-29T10-10-17Z.json`

Do not use NOAA SWPC rolling Kp, Dst, GOES X-ray, or daily F10.7 feeds for historical backtests when the archive sources above are available.

## Next Work

Build small archive connectors for:

* GFZ Kp/Ap text
* Kyoto Dst monthly text
* NOAA/NCEI GOES XRS NetCDF or legacy full/avg text
* Space Weather Canada daily F10.7 text
