---
name: elfquake-data-refresh
description: Use when running ELFQuake prospective real-data refresh scripts, including relabeling matured VLF rows and rebuilding real model input artifacts.
---

# ELFQuake Data Refresh

Run from the repository root. Keep data Italy-scoped, preserve raw records, normalize timestamps to UTC, and do not claim prediction value from refreshed data alone.

## Usual Order

1. Refresh real captures and prospective labels:

```sh
./refresh-prospective-labels.sh
```

2. Rebuild real model inputs:

```sh
./prepare-real-model-inputs.sh
```

3. Check readiness JSON before training real models:

```sh
rg -n "status|labeled_row_count|positive_count|negative_count" data/derived/models/*real_vlf_aligned_windows.readiness.json data/derived/multimodal/*labeled.readiness.json
```

Only run real training when readiness shows both positive and negative examples. If class variation is absent, report the blocker and keep collecting data.

## Useful Follow-ups

Use these only when relevant:

```sh
./materialize-real-vlf-sequence.sh
./train-real-tabular-model.sh
```

After meaningful count or status changes, update `docs/report.md` and `docs/next-actions.md` concisely.
