# Event Schema

First normalized earthquake event schema for Italy-scoped data.

## Required Fields

| Field | Type | Notes |
| --- | --- | --- |
| `event_id` | string | Stable source event identifier |
| `source` | string | Example: `ingv_fdsn_event` |
| `event_time_utc` | datetime | UTC timestamp |
| `latitude` | number | Decimal degrees |
| `longitude` | number | Decimal degrees |
| `depth_km` | number | Kilometers below surface |
| `magnitude` | number | Numeric magnitude |
| `magnitude_type` | string | Preserve source value |
| `italy_region` | string | Derived region label or `unknown` |
| `raw_uri` | string | Source request or record URI |
| `ingested_at_utc` | datetime | Pipeline ingestion time |

## Optional Fields

| Field | Type | Notes |
| --- | --- | --- |
| `time_uncertainty_s` | number | Source uncertainty when available |
| `location_uncertainty_km` | number | Source uncertainty when available |
| `magnitude_uncertainty` | number | Source uncertainty when available |
| `quality` | string | Source quality flag |

## Rules

* Store raw records unchanged before normalization.
* Normalize timestamps to UTC and distances to kilometers.
* Preserve source identifiers and uncertainty fields.
* Exclude records outside the Italy filter from normalized event tables.
