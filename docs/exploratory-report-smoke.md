# Smoke Exploratory Report

First data usability report for the INGV Italy smoke dataset.

## Dataset

* raw: `data/raw/ingv/events_italy_2026-06-22_2026-06-29.txt`
* normalized: `data/derived/ingv/events_italy_2026-06-22_2026-06-29.normalized.csv`
* Central Italy subset: `data/derived/ingv/events_central_italy_2026-06-22_2026-06-29.normalized.csv`
* source: INGV FDSN event text export
* event rows: `26`
* Central Italy rows: `4`

## Counts

Daily event counts:

| Date | Events |
| --- | ---: |
| 2026-06-22 | 1 |
| 2026-06-23 | 0 |
| 2026-06-24 | 5 |
| 2026-06-25 | 4 |
| 2026-06-26 | 5 |
| 2026-06-27 | 5 |
| 2026-06-28 | 4 |
| 2026-06-29 | 2 |

Regional labels:

| Region | Events |
| --- | ---: |
| `central_italy` | 4 |
| `unknown` | 22 |

## Distributions

Magnitude counts:

| Magnitude | Events |
| ---: | ---: |
| 2.0 | 5 |
| 2.1 | 4 |
| 2.2 | 3 |
| 2.3 | 2 |
| 2.4 | 4 |
| 2.5 | 1 |
| 2.6 | 1 |
| 2.7 | 1 |
| 3.0 | 2 |
| 3.2 | 2 |
| 3.6 | 1 |

Depth buckets:

| Depth km | Events |
| --- | ---: |
| `0-10` | 13 |
| `>10-50` | 6 |
| `>50-150` | 5 |
| `>150` | 2 |

## Quality Notes

* Required normalized fields are present for all rows.
* No duplicate event IDs were observed.
* Coordinates are inside the Italy smoke bounding box.
* Timestamps were normalized with `Z`; source timestamps include fractional seconds.
* Raw contributor fields are often empty, but they are not required for the normalized schema.

## Connector Notes

The text export is usable for the smoke dataset. `format=geojson`, narrow Central Italy bounding boxes, and some 2016 historical requests returned server errors during source testing.

## Conclusion

The smoke dataset is usable for normalization and pipeline checks. It is too small and too recent for meaningful model evaluation; use it to define baseline inputs before expanding historical coverage.
