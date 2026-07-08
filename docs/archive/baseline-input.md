# Baseline Input

Input table for the naive historical-rate baseline.

## Source

Use the normalized Central Italy subset:

`data/derived/ingv/events_central_italy_2026-06-22_2026-06-29.normalized.csv`

## Output

`data/derived/baseline/central_italy_7d_m3_smoke.input.csv`

## Schema

| Field | Type | Notes |
| --- | --- | --- |
| `window_start_utc` | datetime | Inclusive window start |
| `window_end_utc` | datetime | Exclusive window end |
| `region` | string | `central_italy` |
| `event_count` | integer | Events in the window |
| `target_magnitude_min` | number | `3.0` |
| `target_event_count` | integer | Events with magnitude `>= 3.0` |
| `target_occurred` | integer | `1` if target event count is positive, else `0` |

## Smoke Window

Use one 7-day window:

* start: `2026-06-22T00:00:00Z`
* end: `2026-06-29T23:59:59Z`
* region: `central_italy`

## Limitation

The smoke subset contains `4` Central Italy events and `0` events with magnitude `>= 3.0`. This is enough to test table shape, but not enough to evaluate a historical-rate model.
