# Derived Datasets

Current derived smoke datasets:

| File | Rows | Notes |
| --- | ---: | --- |
| `data/derived/ingv/events_italy_2026-06-22_2026-06-29.normalized.csv` | 26 | Normalized Italy smoke dataset |
| `data/derived/ingv/events_central_italy_2026-06-22_2026-06-29.normalized.csv` | 4 | Local Central Italy subset |

The Central Italy subset uses the rule in [Normalization](normalization.md). These files are derived from the raw INGV smoke export and should be regenerated, not hand-edited, once ingestion code exists.
