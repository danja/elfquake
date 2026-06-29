# Feature Extraction

Feature builders are stdlib-only stubs under `src/elfquake/features`.

Current outputs:

* `data/derived/multimodal/cumiana_vlf_2026-06-29T0900_1000.features.csv`
* `data/derived/multimodal/astronomy_2026-06-29T0900_1015.features.csv`

Current VLF fields cover capture availability, latest `Last-Modified`, byte count, JPEG detection, and stale/missing flags.

Current astronomy fields cover capture availability, source coverage, next USNO moon phase, and latest available monthly F10.7 value.

These are schema and provenance features only. Add signal features after capture cadence and source coverage are stable.
