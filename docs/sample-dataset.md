# Sample Dataset

Use a small, reproducible pilot before building general ingestion.

## Smoke Scope

This has been built once; see [Smoke Dataset](smoke-dataset.md).

* region: Italy bounding box
* time range: recent 7-day window
* event source: INGV event text export
* spatial filter: latitude `35` to `48`, longitude `6` to `19`
* minimum fields: all required fields from [Event Schema](event-schema.md)

## Pilot Scope

* region: Central Italy
* time range: choose after source access is proven for the requested historical period
* event source: INGV event catalog
* spatial filter: derive Central Italy records from the Italy smoke dataset first
* minimum fields: all required fields from [Event Schema](event-schema.md)

## Acceptance Criteria

* Raw source records are stored separately from normalized records.
* Every normalized record has UTC time, coordinates, depth, magnitude, source, and provenance.
* The dataset includes row counts, date range, magnitude range, and missing-field counts.
* The build can be repeated from documented source requests.

## Exclusions

Do not include VLF, waveform, or astronomical features in this seismic smoke dataset. Evaluate those sources separately and align them through the multimodal window schema.
