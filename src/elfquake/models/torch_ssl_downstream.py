"""Downstream probes and missing-modality checks for self-supervised encoders."""

from __future__ import annotations

import random

from elfquake.models.temporal_holdout import _best_threshold, _metrics, _predictions
from elfquake.models.torch_multimodal_data import build_window_batch


SYNTHETIC_MODALITIES = (
    "synthetic_direct_avalanche",
    "synthetic_piezo_vlf",
    "synthetic_summary",
)
DOWNSTREAM_CONFIGS = {
    "full": SYNTHETIC_MODALITIES,
    "piezo_vlf_only": ("synthetic_piezo_vlf",),
}
MISSING_MODALITY_EVALUATIONS = {
    "full": set(),
    "direct_avalanche_only": {"synthetic_piezo_vlf", "synthetic_summary"},
    "piezo_vlf_only": {"synthetic_direct_avalanche", "synthetic_summary"},
    "summary_only": {"synthetic_direct_avalanche", "synthetic_piezo_vlf"},
    "without_direct_avalanche": {"synthetic_direct_avalanche"},
    "without_piezo_vlf": {"synthetic_piezo_vlf"},
    "without_summary": {"synthetic_summary"},
}


def train_downstream(
    model,
    *,
    train_refs,
    train_labels,
    test_refs,
    test_labels,
    sequences,
    normalizations,
    modalities,
    lookback_steps,
    epochs,
    learning_rate,
    batch_size,
    modality_dropout_probability,
    freeze_backbone,
    optimizer_parameters=None,
    trainable_scope="all",
    seed,
    torch,
):
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    parameters = optimizer_parameters or (model.occurrence_head.parameters() if freeze_backbone else model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    positives = sum(train_labels)
    negatives = len(train_labels) - positives
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32))
    indices = list(range(len(train_refs)))
    first_loss = None
    last_loss = 0.0
    for _ in range(epochs):
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        loss_sum = 0.0
        if freeze_backbone:
            model.eval()
            model.occurrence_head.train()
        else:
            model.train()
        for start in range(0, len(indices), batch_size):
            selected = permutation[start:start + batch_size]
            refs = [train_refs[index] for index in selected]
            labels = torch.tensor([train_labels[index] for index in selected], dtype=torch.float32).unsqueeze(1)
            inputs, _, observed = build_window_batch(
                refs,
                sequences,
                modalities=modalities,
                lookback_steps=lookback_steps,
                normalizations=normalizations,
                torch=torch,
            )
            dropped = _supervised_dropout(modalities, modality_dropout_probability, rng)
            optimizer.zero_grad()
            loss = criterion(model(inputs, observed, dropped_modalities=dropped), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(selected)
        last_loss = loss_sum / len(indices)
        if first_loss is None:
            first_loss = last_loss
    train_probabilities = _probabilities(
        model,
        train_refs,
        sequences,
        normalizations,
        modalities,
        lookback_steps,
        batch_size,
        set(),
        torch,
    )
    threshold = _best_threshold(train_probabilities, train_labels)
    evaluations = {}
    for name, dropped in _missing_modality_evaluations(modalities).items():
        probabilities = _probabilities(
            model,
            test_refs,
            sequences,
            normalizations,
            modalities,
            lookback_steps,
            batch_size,
            dropped,
            torch,
        )
        evaluations[name] = {
            "dropped_modalities": sorted(dropped),
            "default_metrics": _metrics(_predictions(probabilities, threshold=0.5), test_labels),
            "calibrated_metrics": _metrics(_predictions(probabilities, threshold=threshold), test_labels),
        }
    return {
        "modalities": list(modalities),
        "freeze_backbone": freeze_backbone,
        "trainable_scope": trainable_scope,
        "first_train_loss": round(first_loss or 0.0, 8),
        "last_train_loss": round(last_loss, 8),
        "calibrated_threshold": round(threshold, 6),
        "train_metrics": _metrics(_predictions(train_probabilities, threshold=threshold), train_labels),
        "evaluations": evaluations,
    }


def summarize_downstream_runs(runs: list[dict[str, object]], regimes: tuple[str, ...]) -> dict[str, object]:
    summary = {}
    for regime in regimes:
        selected = [run for run in runs if run["regime"] == regime]
        if not selected:
            continue
        downstream_models = {}
        for config_name in DOWNSTREAM_CONFIGS:
            probe = [_balanced_accuracy(run, config_name, "linear_probe") for run in selected]
            fine = [_balanced_accuracy(run, config_name, "fine_tune") for run in selected]
            evaluation_names = selected[0]["downstream_models"][config_name]["fine_tune"]["evaluations"]
            evaluations = {
                name: _distribution([
                    float(run["downstream_models"][config_name]["fine_tune"]["evaluations"][name]["calibrated_metrics"]["balanced_accuracy"])
                    for run in selected
                ])
                for name in evaluation_names
            }
            downstream_models[config_name] = {
                "linear_probe_balanced_accuracy": _distribution(probe),
                "fine_tune_balanced_accuracy": _distribution(fine),
                "fine_tune_missing_modality_evaluations": evaluations,
            }
        summary[regime] = {"run_count": len(selected), "downstream_models": downstream_models}
    random_models = summary.get("random_init", {}).get("downstream_models", {})
    for item in summary.values():
        for config_name, model_summary in item["downstream_models"].items():
            random_mean = random_models.get(config_name, {}).get("fine_tune_balanced_accuracy", {}).get("mean")
            if random_mean is not None:
                model_summary["fine_tune_mean_gain_over_random"] = round(float(model_summary["fine_tune_balanced_accuracy"]["mean"]) - float(random_mean), 8)
    return summary


def _probabilities(model, refs, sequences, normalizations, modalities, lookback, batch_size, dropped, torch):
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(refs), batch_size):
            batch_refs = refs[start:start + batch_size]
            inputs, _, observed = build_window_batch(
                batch_refs,
                sequences,
                modalities=modalities,
                lookback_steps=lookback,
                normalizations=normalizations,
                torch=torch,
            )
            result.extend(torch.sigmoid(model(inputs, observed, dropped_modalities=dropped)).squeeze(1).tolist())
    return [float(value) for value in result]


def _supervised_dropout(modalities: tuple[str, ...], probability: float, rng: random.Random) -> set[str]:
    if len(modalities) < 2 or rng.random() >= probability:
        return set()
    return {rng.choice(modalities)}


def _missing_modality_evaluations(modalities: tuple[str, ...]) -> dict[str, set[str]]:
    if modalities == SYNTHETIC_MODALITIES:
        return MISSING_MODALITY_EVALUATIONS
    return {
        "trained_input": set(),
        **{
            f"without_{_modality_slug(modality)}": {modality}
            for modality in modalities
        },
    }


def _modality_slug(modality: str) -> str:
    return modality.removeprefix("synthetic_")


def _balanced_accuracy(run: dict[str, object], config_name: str, stage: str) -> float:
    evaluation_name = "full" if config_name == "full" else "trained_input"
    return float(run["downstream_models"][config_name][stage]["evaluations"][evaluation_name]["calibrated_metrics"]["balanced_accuracy"])


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(sum(values) / len(values), 8),
        "min": round(min(values), 8),
        "max": round(max(values), 8),
    }
