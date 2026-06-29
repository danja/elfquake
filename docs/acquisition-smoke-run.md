# Acquisition Smoke Run

Smoke run date: `2026-06-29`.

## Results

| Source | Command scope | Stored payload | Status |
| --- | --- | --- | --- |
| USNO moon phases | `usno_moon_phases` only | `data/raw/astronomy/captures/2026-06-29/usno_moon_phases_2026-06-29T09-56-53Z.json` | HTTP 200 |
| Cumiana VLF | `last_E_VLF` only | `data/raw/vlf/cumiana/captures/2026-06-29/last_E_VLF_2026-06-29T09-45-00Z.jpg` | HTTP 200 |
| INGV events | Italy, `2026-06-22` to `2026-06-29` | `data/raw/ingv/events_italy_2026-06-22_2026-06-29_2026-06-29T09-58-18Z.txt` | HTTP 200 |

## Notes

* Each payload has a sibling `.metadata.json` file.
* INGV accepted timestamps without trailing `Z` in the query string.
* The VLF filename uses the source `Last-Modified` timestamp.
* The INGV payload has `27` lines: one header plus `26` events.
