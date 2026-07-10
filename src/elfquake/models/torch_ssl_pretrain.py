"""Masked-patch self-supervision for the multimodal Transformer."""

from __future__ import annotations

import random
from dataclasses import dataclass

from elfquake.models.torch_multimodal_data import (
    ModalitySequence,
    Normalization,
    WindowRef,
    build_window_batch,
)
from elfquake.models.torch_multimodal_encoder import patchify


@dataclass(frozen=True)
class PretrainTask:
    name: str
    modalities: tuple[str, ...]
    train_refs: tuple[WindowRef, ...]
    test_refs: tuple[WindowRef, ...]


def pretrain_masked_patches(
    model: object,
    *,
    tasks: list[PretrainTask],
    sequences: dict[tuple[str, str], ModalitySequence],
    normalizations: dict[str, Normalization],
    lookback_steps: int,
    patch_steps: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    mask_probability: float,
    modality_dropout_probability: float,
    max_windows_per_domain: int,
    balance_domains: bool,
    seed: int,
    torch: object,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    prepared = _prepared_tasks(
        tasks,
        max_windows_per_domain=max_windows_per_domain,
        balance_domains=balance_domains,
        rng=rng,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    first_loss = None
    last_loss = 0.0
    for _ in range(epochs):
        epoch_batches = []
        for task, refs in prepared:
            shuffled = list(refs)
            rng.shuffle(shuffled)
            epoch_batches.extend((task, shuffled[start:start + batch_size]) for start in range(0, len(shuffled), batch_size))
        rng.shuffle(epoch_batches)
        loss_sum = 0.0
        row_count = 0
        model.train()
        for task, refs in epoch_batches:
            inputs, targets, observed = build_window_batch(
                refs,
                sequences,
                modalities=task.modalities,
                lookback_steps=lookback_steps,
                normalizations=normalizations,
                torch=torch,
            )
            corruption = _corruption_masks(
                observed,
                patch_steps=patch_steps,
                probability=mask_probability,
                generator=generator,
                torch=torch,
            )
            dropped = _dropped_modalities(
                task.modalities,
                probability=modality_dropout_probability,
                rng=rng,
            )
            optimizer.zero_grad()
            reconstruction = model.reconstruct(
                inputs,
                observed,
                corruption_masks=corruption,
                dropped_modalities=dropped,
            )
            loss = _reconstruction_loss(
                reconstruction,
                targets,
                observed,
                corruption,
                dropped_modalities=dropped,
                patch_steps=patch_steps,
                torch=torch,
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(refs)
            row_count += len(refs)
        last_loss = loss_sum / max(1, row_count)
        if first_loss is None:
            first_loss = last_loss
    evaluations = {
        task.name: evaluate_masked_reconstruction(
            model,
            refs=list(task.test_refs),
            sequences=sequences,
            normalizations=normalizations,
            modalities=task.modalities,
            lookback_steps=lookback_steps,
            patch_steps=patch_steps,
            batch_size=batch_size,
            mask_probability=mask_probability,
            seed=seed + 10_000,
            torch=torch,
        )
        for task in tasks
    }
    return {
        "first_train_loss": round(first_loss or 0.0, 8),
        "last_train_loss": round(last_loss, 8),
        "epochs": epochs,
        "task_train_windows": {task.name: len(refs) for task, refs in prepared},
        "evaluations": evaluations,
    }


def evaluate_masked_reconstruction(
    model: object,
    *,
    refs: list[WindowRef],
    sequences: dict[tuple[str, str], ModalitySequence],
    normalizations: dict[str, Normalization],
    modalities: tuple[str, ...],
    lookback_steps: int,
    patch_steps: int,
    batch_size: int,
    mask_probability: float,
    seed: int,
    torch: object,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    squared_sum = zero_sum = last_sum = 0.0
    selected_count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(refs), batch_size):
            batch_refs = refs[start:start + batch_size]
            inputs, targets, observed = build_window_batch(
                batch_refs,
                sequences,
                modalities=modalities,
                lookback_steps=lookback_steps,
                normalizations=normalizations,
                torch=torch,
            )
            corruption = _corruption_masks(
                observed,
                patch_steps=patch_steps,
                probability=mask_probability,
                generator=generator,
                torch=torch,
            )
            reconstructed = model.reconstruct(inputs, observed, corruption_masks=corruption)
            for modality in modalities:
                target = patchify(targets[modality], patch_steps=patch_steps, torch=torch)
                present = patchify(observed[modality].float(), patch_steps=patch_steps, torch=torch).bool()
                selected = present & corruption[modality].unsqueeze(2)
                previous = torch.cat([torch.zeros_like(target[:, :1, :]), target[:, :-1, :]], dim=1)
                squared_sum += float(((reconstructed[modality] - target) ** 2)[selected].sum().item())
                zero_sum += float((target ** 2)[selected].sum().item())
                last_sum += float(((previous - target) ** 2)[selected].sum().item())
                selected_count += int(selected.sum().item())
    divisor = max(1, selected_count)
    mse = squared_sum / divisor
    zero_mse = zero_sum / divisor
    last_mse = last_sum / divisor
    return {
        "window_count": len(refs),
        "selected_value_count": selected_count,
        "masked_mse": round(mse, 8),
        "zero_baseline_mse": round(zero_mse, 8),
        "last_patch_baseline_mse": round(last_mse, 8),
        "beats_zero_baseline": mse < zero_mse,
        "beats_last_patch_baseline": mse < last_mse,
    }


def _prepared_tasks(
    tasks: list[PretrainTask],
    *,
    max_windows_per_domain: int,
    balance_domains: bool,
    rng: random.Random,
) -> list[tuple[PretrainTask, tuple[WindowRef, ...]]]:
    cap = min((len(task.train_refs) for task in tasks), default=0) if balance_domains and len(tasks) > 1 else 0
    prepared = []
    for task in tasks:
        refs = list(task.train_refs)
        rng.shuffle(refs)
        requested = len(refs)
        if cap:
            requested = min(requested, cap)
        if max_windows_per_domain:
            requested = min(requested, max_windows_per_domain)
        prepared.append((task, tuple(refs[:requested])))
    return prepared


def _corruption_masks(
    observed: dict[str, object],
    *,
    patch_steps: int,
    probability: float,
    generator: object,
    torch: object,
) -> dict[str, object]:
    result = {}
    for modality, mask in observed.items():
        patch_present = patchify(mask.float(), patch_steps=patch_steps, torch=torch).bool().any(dim=2)
        selected = (torch.rand(patch_present.shape, generator=generator) < probability) & patch_present
        for row_index in range(len(selected)):
            if not bool(selected[row_index].any()) and bool(patch_present[row_index].any()):
                first = int(torch.nonzero(patch_present[row_index], as_tuple=False)[0].item())
                selected[row_index, first] = True
        result[modality] = selected
    return result


def _dropped_modalities(
    modalities: tuple[str, ...],
    *,
    probability: float,
    rng: random.Random,
) -> set[str]:
    if len(modalities) < 2 or rng.random() >= probability:
        return set()
    return {rng.choice(modalities)}


def _reconstruction_loss(
    reconstruction: dict[str, object],
    targets: dict[str, object],
    observed: dict[str, object],
    corruption: dict[str, object],
    *,
    dropped_modalities: set[str],
    patch_steps: int,
    torch: object,
):
    losses = []
    for modality, predicted in reconstruction.items():
        target = patchify(targets[modality], patch_steps=patch_steps, torch=torch)
        present = patchify(observed[modality].float(), patch_steps=patch_steps, torch=torch).bool()
        selected = present if modality in dropped_modalities else present & corruption[modality].unsqueeze(2)
        if bool(selected.any()):
            losses.append(((predicted - target) ** 2)[selected].mean())
    if not losses:
        raise ValueError("self-supervised batch selected no observed reconstruction targets")
    return torch.stack(losses).mean()
