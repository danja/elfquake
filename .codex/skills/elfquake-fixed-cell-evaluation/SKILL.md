---
name: elfquake-fixed-cell-evaluation
description: Use when evaluating the Italy fixed-cell spatial target design, running the four standing controls per capture era, or reporting held-out balanced accuracy for VLF/astronomy ablations.
---

# ELFQuake Fixed-Cell Evaluation

Run from the repository root with the venv available. Every step below is
required; skipping one has produced a reportable-looking number that turned out
to be an artifact at least five times (see `MISTAKES.md` and
[Target Design](../../docs/target-design.md)).

## Refresh first, always

The live `elfquake-prospective.timer` fires every 30 minutes but **does not
fetch INGV events**. It only updates VLF image features and the window table
against whatever event catalog is already on disk. Between manual refreshes the
catalog falls behind while anchors keep accumulating, and anchors in the gap get
labeled negative because no event can be found in a catalog that stopped.

```sh
./scripts/refresh-prospective-labels.sh
```

Confirm the catalog actually advanced before trusting anything downstream:

```sh
PYTHONPATH=src .venv/bin/python -c "
import csv
from pathlib import Path
rows = list(csv.DictReader(Path('data/derived/ingv/events_italy_prospective.current.normalized.csv').open(newline='', encoding='utf-8')))
print('events', len(rows))
print('last event  ', max(r['event_time_utc'] for r in rows))
print('last ingested', max(r['ingested_at_utc'] for r in rows))
"
```

`last ingested` is the catalog's coverage end. If it is hours or days behind
now, stop and refresh; do not evaluate.

## Pin the run

Results depend on wall-clock time, so pin both stamps and record them beside any
number you report. Without this the run is not reproducible.

```sh
export ELFQ_AS_OF=2026-08-21T11:22:36Z
export ELFQ_CATALOG_END=2026-08-21T11:10:37Z   # max ingested_at_utc
```

## Build the scoped table

The horizon is fixed when the prospective table is built, not when targets are
labeled, so a different horizon needs a separately scoped table. Never repoint
the live 7-day table or the systemd unit.

```sh
PYTHONPATH=src .venv/bin/python -m elfquake.cli build-prospective-vlf-windows \
  --events data/derived/ingv/events_italy_prospective.current.normalized.csv \
  --vlf-metadata-root data/raw/vlf/cumiana \
  --space-weather-root data/derived/astronomy \
  --region-id all_italy --lookback-hours 24 --horizon-days 1 \
  --target-magnitude-min 2.0 \
  --out data/derived/multimodal/all_italy.prospective_vlf_image_windows.h1m20.csv

INPUT=data/derived/multimodal/all_italy.prospective_vlf_image_windows.h1m20.csv \
OUT=data/derived/multimodal/all_italy.spatial_vlf_image_windows.h1m20.labeled.csv \
AS_OF="$ELFQ_AS_OF" CATALOG_END="$ELFQ_CATALOG_END" \
TARGET_MAGNITUDE_MIN=2.0 CELL_DEGREES=1.5 \
  ./scripts/build-italy-spatial-vlf-targets.sh

./scripts/materialize-real-vlf-sequence.sh

REBUILD_INPUTS=0 \
INPUT=data/derived/multimodal/all_italy.spatial_vlf_image_windows.h1m20.labeled.csv \
PREFIX=data/derived/models/all_italy_spatial_vlf_image_windows_h1m20 \
  ./scripts/prepare-italy-spatial-model-inputs.sh
```

`REBUILD_INPUTS=0` is correct here only because the two upstream rebuilds were
just run explicitly above. Do not set it to skip a refresh.

## Split by capture era

Item 96 forbids pooling across the collector outage: the palette/gain shift puts
the eras' VLF features on different scales. Every evaluation runs inside one
era.

```sh
INPUT=data/derived/models/all_italy_spatial_vlf_image_windows_h1m20_aligned_windows.csv \
OUT_DIR=data/derived/reports/italy_capture_era_shift_h1m20 \
  ./scripts/diagnose-vlf-capture-era-shift.sh
```

Per-era CSVs land in `.../eras/`. Only `kind: era` entries are usable; isolated
single-anchor eras are not.

## Run all four standing controls, per era

```sh
ERA_DIR=data/derived/reports/italy_capture_era_shift_h1m20/eras
OUT=data/derived/reports/italy_h1m20
mkdir -p "$OUT"

for ERA in era_0 era_3; do
  INPUT="$ERA_DIR/$ERA.csv" OUT="$OUT/$ERA.grouped_holdout.json" \
  STRATIFY_FIELD=target_cell_id ./scripts/evaluate-italy-spatial-baseline.sh

  INPUT="$ERA_DIR/$ERA.csv" OUT_DIR="$OUT" \
  STRIPPED="$OUT/$ERA.no_cell_coordinates.csv" \
  OUT="$OUT/$ERA.coordinate_control.json" \
  STRATIFY_FIELD=target_cell_id ./scripts/evaluate-italy-spatial-coordinate-control.sh

  INPUT="$ERA_DIR/$ERA.csv" OUT_DIR="$OUT/permutation_$ERA" \
  EPOCHS=600 STRATIFY_FIELD=target_cell_id \
    ./scripts/evaluate-italy-spatial-permutation-controls.sh
done
```

`STRATIFY_FIELD=target_cell_id` is what adds the cell-stratified metric and the
`stratum_base_rate` control, so it is not optional. Match `EPOCHS` between the
permutation controls and the real run; item 93 was distorted by comparing 100
epochs against 600.

The permutation controls are slow (roughly 5 minutes per seed on `era_3`). Run
them in the background rather than blocking.

## Reading the output

```
stratum base-rate control: 0.533749 plain, 0.5 stratified
held-out label transitions: 19
seismic_only: 0.523786 plain, 0.550318 stratified (8/19 strata, 19 transitions)
```

* **Read the stratified number, not the pooled one.** Pooled balanced accuracy
  cannot separate "predicts when" from "knows which cells are seismically
  active". A cell-constant predictor scores exactly `0.5` stratified.
* **`stratum_base_rate` must come out at `0.500000` stratified.** If it does
  not, the metric is broken and nothing else in the report is readable.
* **Never quote a score without its transition count.** Target windows overlap
  by more than 99%, so rows are near-copies and the row count overstates the
  evidence by orders of magnitude. Transitions are the sample size.
* **Below roughly a hundred held-out transitions per era, report the count and
  stop.** Differences between ablations at that scale are coin flips; the
  per-cell spread on 19 transitions runs `0.30`–`0.85`.

## Design diagnostics

To measure what a candidate design could express before fitting anything:

```sh
INPUT=data/derived/multimodal/all_italy.prospective_vlf_image_windows.h1m20.csv \
HORIZON_DAYS="1 2 3 7" CELL_DEGREES="0.75 1.5 3.0" TARGET_MAGNITUDE_MIN="2.0 2.5" \
OUT=data/derived/reports/italy_target_design/sweep.json \
  ./scripts/diagnose-spatial-target-design.sh
```

Run it **per era**, not on the whole record. A whole-record transition count
answers a question no evaluation asks, because every evaluation is split by era
first. Quoting the whole-record figure for a per-era run overstated the
available evidence by roughly 2.5x on 2026-08-21.

## Do not

* Do not claim VLF or astronomy adds value without an ablation that separates
  from `stratum_base_rate` on the stratified metric, on held-out data, with the
  transition count reported.
* Do not pool eras.
* Do not repoint the live prospective table or the systemd unit to a new
  horizon; use a scoped `*.h<days>m<mag>.*` path.
