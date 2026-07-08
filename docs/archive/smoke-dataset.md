# Smoke Dataset

First reproducible raw dataset pull for Italy.

## File

`data/raw/ingv/events_italy_2026-06-22_2026-06-29.txt`

## Source Request

See [Connector Notes](connector-notes.md) for the exact INGV text export URL.

## Scope

* source: INGV FDSN event text export
* time range: `2026-06-22T00:00:00` to `2026-06-29T23:59:59`
* latitude: `35` to `48`
* longitude: `6` to `19`
* magnitude: `2` to `10`
* row count: `26` event rows

## Observed Schema

```text
#EventID|Time|Latitude|Longitude|Depth/Km|Author|Catalog|Contributor|ContributorID|MagType|Magnitude|MagAuthor|EventLocationName|EventType
```

## Next Use

Use this file to define normalization and derive a Central Italy subset locally. Keep the raw file unchanged.
