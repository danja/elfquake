# Normalization

Normalize INGV text exports into a derived dataset. Keep raw files unchanged.

## Input

`data/raw/ingv/events_italy_2026-06-22_2026-06-29.txt`

## Output

`data/derived/ingv/events_italy_2026-06-22_2026-06-29.normalized.csv`

## Field Mapping

| Raw field | Normalized field | Rule |
| --- | --- | --- |
| `EventID` | `event_id` | string |
| `Time` | `event_time_utc` | append `Z`; source time is treated as UTC |
| `Latitude` | `latitude` | decimal degrees |
| `Longitude` | `longitude` | decimal degrees |
| `Depth/Km` | `depth_km` | kilometers |
| `MagType` | `magnitude_type` | preserve source value |
| `Magnitude` | `magnitude` | number |
| `EventLocationName` | `event_location_name` | preserve source value |
| `EventType` | `event_type` | preserve source value |

Add derived fields:

* `source`: `ingv_fdsn_event_text`
* `italy_region`: `central_italy` or `unknown`
* `raw_file`: source raw file path
* `ingested_at_utc`: source pull timestamp
* `raw_uri`: source request URL from [Connector Notes](connector-notes.md)

## Central Italy Rule

For the smoke dataset, assign `central_italy` when:

* latitude is from `41.5` to `43.5`
* longitude is from `12.0` to `14.5`

Use `unknown` otherwise.

## Checks

* row count matches raw event rows
* required normalized fields are present
* coordinates stay inside the Italy bounding box
* raw file is not modified
