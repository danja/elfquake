---
name: elfquake-synthetic-modeling
description: Use when training, comparing, diagnosing, or selecting ELFQuake synthetic PyTorch models, including multimodal sequence models and missing-modality ablation checks.
---

# ELFQuake Synthetic Modeling

Run from the repository root with the project virtual environment active. Keep models CPU-compatible and preserve swappable model backends.

## Usual Order

1. Train current synthetic baselines:

```sh
./train-synthetic-torch-model.sh
./train-synthetic-sequence-model.sh
```

2. Run sequence comparisons and selection checks:

```sh
./sweep-synthetic-sequence-model.sh
./matched-sequence-comparison.sh
./repeat-sequence-training-seeds.sh
./train-sequence-full-regime.sh
```

3. Summarize and diagnose model outputs:

```sh
./compare-model-runs.sh
./compare-real-synthetic-models.sh
./diagnose-sequence-models.sh
./summarize-sequence-selection.sh
```

4. Check missing-modality behavior:

```sh
./test-sequence-missing-modalities.sh
```

Use an 80 percent training, 20 percent held-out test split unless a script has a documented time-based split. Interpret synthetic runs as engineering validation only; do not imply real earthquake prediction skill.

After material model changes, update `docs/report.md`, `docs/model-candidates.md`, and `docs/next-actions.md` as appropriate.
