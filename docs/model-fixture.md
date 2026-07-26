# Transformer Fixture

## Current Artifact

`data/derived/models/mixed_source_transformer_fixture.alignment.json` records the current CPU fixture inputs:

`./scripts/build-common-transformer-fixture.sh` now produces the first common window table at `data/derived/models/common_transformer_fixture.csv`, together with its readiness report, tensor spec, and materialized masks.

* Synthetic direct avalanche, piezo/VLF-like, and summary sequences from seed 40.
* Italy Cumiana VLF sequence and all-Italy multimodal window tensor.
* Three research-only Japan Moshiri CDF sequence manifests.

The Japan manifests are capture-specific so lookback windows cannot cross month-long gaps. They use `dataset_id` values that match the Japan target table and preserve `research_use_only=1` in their manifests.

## Trainability

The artifact is an alignment and provenance fixture, not yet a single valid mixed-domain training set. Synthetic sequences share one UTC demonstration period; Italy and Japan observations have different coverage and cadence. A model run must therefore either:

1. train separate modality/domain ablations with time-safe splits, or
2. use a dataset builder that creates one row per domain window and emits missing-modality masks for unavailable sources.

The common window table satisfies the tabular interface and is marked `ready_for_smoke_training` with 5,546 rows, 5,508 labeled rows, and 208 numeric features. After the latest Japan refresh, the derived sequence fixture at `data/derived/models/common_transformer_fixture_sequences/fixture.json` contains 88 capture- and dataset-specific manifests covering eight modality groups and 11 dataset IDs. Missing groups are represented by zero-filled channels with zero-valued presence masks, rather than by dropping the modality.

A two-epoch CPU patch-Transformer smoke run is stored at `data/derived/models/common_transformer_fixture_patch_transformer.json`. It completed with 1,072 test rows and calibrated balanced accuracy `0.489320`, below the majority baseline of `0.5`. This is an interface and missing-modality test, not evidence of predictive utility: the fixture combines different time periods and source domains, and its Japan VLF observations are sparse and research-only.

`data/derived/models/common_transformer_fixture.alignment_audit.json` is the alignment gate. The current data contain 5,301 Italy rows with seismic, Cumiana VLF, and astronomy values; only 3 Japan rows with both seismic and observed Moshiri VLF; and 167 synthetic rows with piezo/VLF-like, direct-avalanche, synthetic-seismic, and summary channels. The 75 other Japan target rows have no observed VLF and must remain masked.

Do not concatenate the raw sequence manifests and treat them as continuous time. The alignment report currently flags the required time-scale mapping and source-specific gaps.

## Remaining Gate

Before mixed real/synthetic Transformer evaluation, materialize a common window table with:

* stable `dataset_id` and `region_id`;
* identical UTC window and target fields;
* synthetic, Italy, and Japan modality references;
* astronomy features or an explicit missing mask;
* chronological train, validation, and test partitions;
* modality-presence counts and research-use provenance.

The first valid Japan-only Transformer check requires a later Japan CDF capture in the held-out period. Until then, Japan sequence training is an interface test only. The common sequence fixture is also windowized rather than raw continuous VLF data; it checks the shared model contract and masking behavior, not scientific alignment.
