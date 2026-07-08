# Multimodal Window Schema and Feature Extraction

This document outlines the prediction window schema, alignment grain, feature groups, data leakage constraints, and the commands and outputs for building the design matrix.

---

## 1. Window Grain and Target Specification

Training rows represent one geographical region over a specific time/prediction window.
*   **Grain**: One row per `region_id`, `window_start_utc`, and `window_end_utc`.
*   **Initial Smoke Target**: Predict whether at least one event with magnitude `>= 3.0` occurs in `central_italy` within the next `7` days.

---

## 2. Row Schema Definition

### Required Core Fields
| Field | Type | Description / Constraints |
| --- | --- | --- |
| `window_id` | string | Stable hash/slug uniquely identifying the region and time bounds. |
| `region_id` | string | Named Italy region (e.g., `central_italy`) or grid cell. |
| `window_start_utc`| datetime | Inclusive start of the feature/lookback window. |
| `window_end_utc` | datetime | Exclusive end of the feature window / start of the target window. |
| `lookback_start_utc`| datetime | Earliest timestamp allowed for feature calculations in this row. |
| `target_start_utc` | datetime | Equals `window_end_utc` (no gap between features and prediction). |
| `target_end_utc` | datetime | End of the prediction window (e.g. `target_start_utc + 7 days`). |
| `target_event_count`| integer | Number of matching events occurring in `[target_start_utc, target_end_utc)`. |
| `target_occurred` | integer | `1` if `target_event_count` > 0, else `0`. |
| `target_status` | string | Maturity state (e.g. `unlabeled_pending_future_events`). |

### Feature Group Prefixes
| Prefix | Modality | Description |
| --- | --- | --- |
| `seismic_` | INGV events | Prior event counts, max magnitude, depth summaries. |
| `vlf_` | Cumiana VLF | Image intensity, cropped frequency bands, streak counts, availability. |
| `astro_` | Astronomy/Space | Kp index stats, daily F10.7 flux, distance in days to moon phases. |
| `quality_` | Quality/Staleness | Missing data indicators, response warnings, stale counts. |

---

## 3. In-Sample Feature Extraction Commands

Feature extractors live in `src/elfquake/features/`. Use the following commands to construct standalone features and combine them into design matrices:

To build cropped VLF visual features from JPEGs:
```sh
PYTHONPATH=src python3 -m elfquake.cli build-vlf-features --metadata data/raw/vlf/cumiana/captures/2026-06-29/last_E_VLF_2026-06-29T09-45-00Z.jpg.metadata.json --window-start 2026-06-29T09:00:00Z --window-end 2026-06-29T10:00:00Z --out data/derived/multimodal/cumiana_vlf_features.csv
```

To build coarse astronomical features from JSON metadata:
```sh
PYTHONPATH=src python3 -m elfquake.cli build-astronomy-features --metadata data/raw/astronomy/captures/2026-06-29/usno_moon_phases_2026-06-29T09-56-53Z.json.metadata.json --metadata data/raw/astronomy/captures/2026-06-29/noaa_solar_cycle_f107_2026-06-29T10-10-17Z.json.metadata.json --window-start 2026-06-29T09:00:00Z --window-end 2026-06-29T10:15:00Z --out data/derived/multimodal/astronomy_features.csv
```

To build a manifest-driven multimodal design matrix:
```sh
PYTHONPATH=src python3 -m elfquake.cli build-multimodal-table --manifest data/derived/multimodal/manifests/central_italy_smoke_windows.csv --out data/derived/multimodal/central_italy_smoke_windows.multimodal.csv
```

To apply target labeling using normalized seismic event logs:
```sh
PYTHONPATH=src python3 -m elfquake.cli label-multimodal-targets --input data/derived/multimodal/central_italy_smoke_windows.multimodal.csv --events data/derived/ingv/events_central_italy.normalized.csv --as-of 2026-06-29T10:15:00Z --out data/derived/multimodal/central_italy_labeled.csv
```

---

## 4. Derived Alignment Datasets and Outputs

The matrix generation writes derived CSV artifacts under `data/derived/multimodal/`. Derived files must be programmatically generated via the commands above, rather than hand-edited:

*   **Seismic Training Windows**: e.g., `central_italy_2026-06-01_2026-06-29.seismic_training_windows.csv`
    *   *Details*: 7-day feature windows, 7-day target windows, M3.0+ target.
*   **Design Matrices**: e.g., `central_italy_2026-06-01_2026-06-29.multimodal_design_matrix.csv`
    *   *Details*: Aligns seismic event count, max magnitude, Kp mean/max, ap mean/max, F10.7 mean, VLF capture counts/features, quality indicators, and target labels.
*   **Labeled Prospective Tables**: e.g., `central_italy.prospective_vlf_image_windows.labeled.csv`
    *   *Details*: Rebuilt using `update-prospective-vlf-table` script on a cron/systemd cadence to process incoming mature windows.

---

## 5. Strict Data Leakage Rules

*   **Temporal Causality**: Feature timestamps must be strictly `< target_start_utc`.
*   **Target Bounding**: Targets must only use events in `[target_start_utc, target_end_utc)`.
*   **VLF Imagery**: Do not use VLF images or metadata files with `Last-Modified` timestamps after `target_start_utc`.
*   **Calendar Boundaries**: Calendar-based astronomical positions (like future full moons) are known, but their calculation or acquisition records must be logged prior to `target_start_utc`.
*   **Missing Features**: Do not drop rows with missing features; instead, represent missing states explicitly as indicator values (e.g. `quality_missing_vlf = 1`) to preserve training alignment.

