# Input Freshness (2026-08-22)

Closes [Next Actions](next-actions.md) item 95(d): the audit of the remaining
scripts for the staleness pattern first recorded in item 89.

## The defect class

A script reads two inputs. One is on a refresh path and is rewritten
regularly; the other is not, and nothing rebuilds it. The run succeeds, the
output carries today's timestamp, and half the answer is weeks old.

It is hard to see because the symptom looks like a virtue. Item 89 stated the
tell plainly, and it is worth repeating: **a score that reproduces to six
decimal places across a data refresh is a staleness signal, not a stability
result.**

The class has now been found four times in this repository:

1. `refresh-prospective-labels.sh` rewrote the date-scoped catalogs but never
   `events_italy_all_available.combined.normalized.csv`, which six downstream
   consumers read. Frozen for a month (item 89).
2. `prepare-italy-spatial-model-inputs.sh` rebuilt its upstream inputs only
   when they were missing, so a refresh reused Aug-4 targets (item 89).
3. `elfquake-prospective.service` fired every 30 minutes and never fetched
   INGV, so the live collector labeled 22 real events as non-events inside a
   held-out partition (item 104).
4. Everything below.

## What this audit found

Modification times as of `2026-08-22T08:48Z`, when the event catalog was
current to the minute.

| Consumer | Stale input | Age | Nothing rebuilds it because |
| --- | --- | --- | --- |
| `analyze-italy-vlf-event-association.sh` | `self_supervised/real_vlf_anomaly_scores.csv` | 13 d | written by `score-real-vlf-anomaly-forecast.sh`, not on any refresh path |
| `report-italy-data-coverage.sh` | same | 13 d | same |
| `trial-weekly-event-forecast.sh` | `*.real_vlf_aligned_windows.csv`, `real_vlf_anomaly_forecast.json` | 15 d, 13 d | written by `prepare-real-model-inputs.sh` |
| `learned-weekly-event-forecast.sh` | same, plus the aligned synthetic windows | 15 d, 46 d | same |
| `run-cross-region-generative-smoke.sh` | `common_transformer_fixture_sequences/*/manifest.json` | 13 d behind the fixture | `materialize-common-transformer-sequences.sh` is a separate manual step |
| five `evaluate-italy-spatial-*.sh` | `*_aligned_windows.csv` | matched here, but ungated | each caller kept the existence-only guard item 89 removed from the callee |

Two of these were live defects rather than latent ones:

* **The transformer trained on dead channels.** The sequence tensors were 13
  days behind the fixture, so the astronomy modality still carried
  `astro_capture_count` and `astro_noaa_solar_cycle_f107_value` — the two
  channels item 99 found useless — while the fixture had held the 19 aligned
  channels since 2026-08-17. Rematerializing raised the astronomy sequence
  from 2 channels to 19. See [Astronomy alignment](astronomy-alignment.md).
* **A frozen question, not a frozen file.** Both weekly forecast scripts
  defaulted `AS_OF_UTC` to the literal `2026-07-08T00:00:00Z`. The catalog was
  fresh, the forecast was for a July week, every run. This is the same defect
  in the argument list rather than in a path, and no file timestamp would ever
  have revealed it.

## The guard

`scripts/lib/staleness.sh`, sourced by the scripts above.

```sh
. "$(dirname "$0")/lib/staleness.sh"
require_fresh_inputs "$EVENTS" "$ANOMALY" "$VLF_WINDOWS"
```

The first argument is the reference — the input the refresh pipeline does keep
current, normally the INGV event catalog. Every later argument is compared
against it by modification time.

* **Warn by default.** An audit run still produces its report, with the caveat
  on stderr where the operator sees it. Failing by default would make a
  one-off diagnostic unusable on a laptop that has not run the full pipeline.
* **`STALE_INPUTS=fail`** turns the warning into exit status `3`, for
  unattended and CI use.
* **It never rebuilds anything.** The artifacts it guards are model outputs
  that cost far more than the report consuming them. Whether to spend that is
  the caller's decision.
* **A missing input is not staleness.** It is a different failure and the
  command that needs the file reports it in its own terms.
* **The listing is capped at 8** (`STALE_INPUTS_MAX_LISTED`), with the true
  count above it. A caller that passes a glob can match dozens of files, and a
  40-line warning trains the reader to skip warnings. This came up immediately:
  `common_transformer_fixture_sequences/` accumulates manifests for datasets
  that have since dropped out of the fixture — 40 orphaned Japan per-file
  datasets from late July — and `materialize-common-transformer-sequences.sh`
  rewrites only the datasets the current fixture contains, so the orphans stay
  behind at their original timestamps forever. Point a guard at the artifacts a
  script actually opens, not at a directory.

The same file carries `catalog_coverage_end`, which reads `coverage_end_utc`
from `data/derived/ingv/catalog_freshness.json`. Coverage is what the catalog
was successfully *asked* for, never wall-clock time and never
`max(ingested_at_utc)` — `combine-normalized-events` deduplicates by
`event_id` and keeps the first occurrence, so a fetch that finds no new events
leaves every ingest stamp untouched and a current catalog looks stale. A quiet
period is data, not an absence of coverage.

Regression cover is in `tests/test_staleness_guard.py`.

## What the guard does not do

It compares modification times. It cannot see a file rewritten with identical
content, an input whose *contents* are stale while its timestamp is fresh, or
the frozen-argument case above. Those need the item-89 tell — watch for a
score that does not move when the data does.
