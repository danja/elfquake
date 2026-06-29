# Data Sources

Initial work should confirm availability, licensing, quality, and timestamp precision before building models.

## Candidate Sources

* INGV earthquake event data: event time, location, depth, magnitude, and uncertainty.
* vlf.it natural radio observations: signal recordings, metadata, station details, and collection notes.
* Astronomical data: lunar phase, Earth cycles, solar activity, and related time indexes.

## Source Checklist

For each source, document:

* access method and update cadence
* license or reuse constraints
* raw format and schema
* timezone and timestamp precision
* spatial coverage
* known gaps and quality warnings

## Normalization

Store raw data unchanged, then derive normalized datasets with explicit units, UTC timestamps, source identifiers, and provenance metadata.
