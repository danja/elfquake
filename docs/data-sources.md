# Data Sources

Initial work should confirm availability, licensing, quality, and timestamp precision for Italy before building models.

## Candidate Sources

* INGV earthquake event data for Italy: event time, location, depth, magnitude, and uncertainty.
* Italy-relevant natural radio observations: signal recordings, metadata, station details, and collection notes.
* Astronomical data applicable to Italy: lunar phase, Earth cycles, solar activity, and related time indexes.

See [Source Inventory](source-inventory.md) for current source status.

## Source Checklist

For each source, document:

* access method and update cadence
* license or reuse constraints
* raw format and schema
* timezone and timestamp precision
* spatial coverage
* Italy boundary or region filter used
* known gaps and quality warnings

## Normalization

Store raw data unchanged, then derive Italy-scoped normalized datasets with explicit units, UTC timestamps, source identifiers, region labels, and provenance metadata.
