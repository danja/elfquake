# Model Scaling Requirements

Purpose: decide when a larger GRU or Transformer-style model is justified.

## Current Scale Checks

Run:

```sh
./estimate-model-scale.sh
```

Current artifacts:

* `data/derived/models/model_scale_synthetic_20000.json`
* `data/derived/models/model_scale_synthetic_balanced_20000.json`
* `data/derived/models/model_scale_all_italy_real_vlf.json`
* `data/derived/models/model_scale_central_italy_real_vlf.json`

## Current Status

| dataset | labeled rows | positives | negatives | sequence features | recommendation |
| --- | ---: | ---: | ---: | ---: | --- |
| synthetic 20000-step | 1005 | 364 | 641 | 168 | tiny synthetic-only patch Transformer is plausible |
| synthetic post-burn-in balanced | 804 | 360 | 444 | 168 | GRU/tabular smoke only |
| all-Italy real VLF-aligned | 23 | 23 | 0 | 50 | do not train supervised model |
| central-Italy real VLF-aligned | 23 | 0 | 23 | 50 | do not train supervised model |

## Gates

Minimum gates before scaling:

* tabular or GRU smoke: at least 500 labeled rows and 50 rows per class
* larger GRU or tiny synthetic Transformer: at least 1000 labeled rows, 100 rows per class, 3 groups, and sequence features
* small synthetic Transformer: at least 5000 labeled rows, 500 rows per class, 5 groups, and sequence features
* real Transformer training: at least 5000 real labeled rows and 500 rows per class

The current real VLF-aligned rows fail the first gate because each scope has only one class.

## CPU-Only Starting Point

If testing a larger synthetic-only model, keep it tiny:

* patch/channel encoder before cross-modality attention
* `d_model=32`
* `layers=2`
* `heads=2`
* `batch_size=32`
* dropout around `0.1`
* strict ablations: seismic-only, VLF-only, direct avalanche only, full

Avoid large attention models and GPU-only dependencies on the current system.

## Current Larger-Model Check

The first larger-model scaffold is implemented as a tiny CPU-only patch Transformer:

* command: `./train-tiny-patch-transformer.sh`
* output: `data/derived/models/tiny_patch_transformer/tiny_patch_transformer_model_run_summary.json`
* split: post-burn-in regime-balanced synthetic rows, 642 train / 162 test
* best calibrated balanced accuracy: `0.637500` for `sequence_piezo_vlf_only`

This is below the current regime-balanced GRU `sequence_full` score of `0.650000`, so it validates the model interface but does not replace the GRU baseline.

## Recommendation

Do not scale real models yet. For synthetic-only engineering, compare the tiny patch Transformer across additional generated seeds before increasing size. Treat any synthetic gain as model-interface evidence, not earthquake prediction evidence.
