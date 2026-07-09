---
name: elfquake-synthetic-modeling
description: Use when training, comparing, diagnosing, or selecting ELFQuake synthetic PyTorch models, including multimodal sequence models and missing-modality ablation checks.
---

# ELFQuake Synthetic Modeling

Run from the repository root with the project virtual environment active. Keep models CPU-compatible and preserve swappable model backends.

## Usual Order

1. Train current synthetic baselines:

```sh
./scripts/train-synthetic-torch-model.sh
./scripts/train-synthetic-sequence-model.sh
```

2. Run sequence comparisons and selection checks:

```sh
./scripts/sweep-synthetic-sequence-model.sh
./scripts/matched-sequence-comparison.sh
./scripts/repeat-sequence-training-seeds.sh
./scripts/train-sequence-full-regime.sh
./scripts/train-sequence-full-balanced.sh
./scripts/train-tiny-patch-transformer.sh
```

3. Summarize and diagnose model outputs:

```sh
./scripts/compare-model-runs.sh
./scripts/compare-real-synthetic-models.sh
./scripts/estimate-model-scale.sh
./scripts/diagnose-sequence-models.sh
./scripts/summarize-sequence-selection.sh
```

4. Check missing-modality behavior:

```sh
./scripts/test-sequence-missing-modalities.sh
```

Use an 80 percent training, 20 percent held-out test split unless a script has a documented time-based split. Interpret synthetic runs as engineering validation only; do not imply real earthquake prediction skill.

After material model changes, update `docs/report.md`, `docs/model-candidates.md`, and `docs/next-actions.md` as appropriate.
