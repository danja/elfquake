# Multimodal Window Schema

Training rows represent one Italy region over one prediction window.

## Grain

Use one row per `region_id`, `window_start_utc`, and `window_end_utc`.

Initial smoke grain:

* `region_id`: `central_italy`
* `window_length`: `7 days`
* `target_magnitude_min`: `3.0`

## Required Fields

| Field | Type | Notes |
| --- | --- | --- |
| `window_id` | string | Stable hash or slug for region and time bounds |
| `region_id` | string | Named Italy region or grid cell |
| `window_start_utc` | datetime | Inclusive |
| `window_end_utc` | datetime | Exclusive |
| `lookback_start_utc` | datetime | Earliest feature timestamp allowed |
| `target_start_utc` | datetime | Usually equal to `window_end_utc` |
| `target_end_utc` | datetime | Future target horizon end |
| `target_event_count` | integer | Events meeting target criteria |
| `target_occurred` | integer | `1` if count is positive |

## Feature Groups

| Prefix | Modality | Examples |
| --- | --- | --- |
| `seismic_` | INGV events and waveforms | prior event counts, max magnitude, depth summaries |
| `vlf_` | Cumiana VLF captures | image availability, band summaries, anomaly scores, stale flags |
| `astro_` | Lunar, solar, geomagnetic | Kp summaries, X-ray flux summaries, lunar phase distance |
| `quality_` | Data quality | missing counts, stale source flags, source coverage |

## Leakage Rules

* Feature timestamps must be `< target_start_utc`.
* Targets must use events in `[target_start_utc, target_end_utc)`.
* Do not use image `Last-Modified` values later than `target_start_utc`.
* Missing source coverage must be represented as features, not silently dropped.
* Validation windows must occur after training windows.

## First Ablations

Evaluate the same windows with:

1. seismic-only features
2. seismic plus VLF features
3. seismic plus astronomical features
4. seismic plus VLF and astronomical features
