"""Reusable modality-adapter patch Transformer for CPU training."""

from __future__ import annotations

import hashlib
import math


def build_multimodal_patch_transformer(
    torch: object,
    *,
    input_sizes: dict[str, int],
    target_sizes: dict[str, int],
    lookback_steps: int,
    patch_steps: int,
    d_model: int,
    layers: int,
    heads: int,
    dropout: float,
    initialization_seed: int | None = None,
):
    if d_model % heads:
        raise ValueError("heads must divide d_model")
    patch_count = math.ceil(lookback_steps / patch_steps)

    class MultimodalPatchTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_steps = patch_steps
            self.patch_count = patch_count
            self.adapters = torch.nn.ModuleDict({
                modality: torch.nn.Linear(size * patch_steps, d_model)
                for modality, size in input_sizes.items()
            })
            self.decoders = torch.nn.ModuleDict({
                modality: torch.nn.Linear(d_model, size * patch_steps)
                for modality, size in target_sizes.items()
            })
            self.modality_embeddings = torch.nn.ParameterDict({
                modality: torch.nn.Parameter(torch.zeros(1, 1, d_model))
                for modality in input_sizes
            })
            self.mask_tokens = torch.nn.ParameterDict({
                modality: torch.nn.Parameter(torch.zeros(1, 1, d_model))
                for modality in input_sizes
            })
            self.missing_tokens = torch.nn.ParameterDict({
                modality: torch.nn.Parameter(torch.zeros(1, 1, d_model))
                for modality in input_sizes
            })
            self.position = torch.nn.Parameter(torch.zeros(1, patch_count, d_model))
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = torch.nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers,
                enable_nested_tensor=False,
            )
            self.occurrence_head = torch.nn.Linear(d_model, 1)
            torch.nn.init.normal_(self.position, mean=0.0, std=0.02)
            for parameters in (self.modality_embeddings, self.mask_tokens, self.missing_tokens):
                for value in parameters.values():
                    torch.nn.init.normal_(value, mean=0.0, std=0.02)

        def encode(
            self,
            inputs: dict[str, object],
            observed: dict[str, object],
            *,
            corruption_masks: dict[str, object] | None = None,
            dropped_modalities: set[str] | None = None,
        ) -> tuple[object, dict[str, slice], object]:
            corruption_masks = corruption_masks or {}
            dropped_modalities = dropped_modalities or set()
            tokens = []
            padding_masks = []
            slices: dict[str, slice] = {}
            offset = 0
            for modality in sorted(inputs):
                patches = patchify(inputs[modality], patch_steps=self.patch_steps, torch=torch)
                observed_patches = patchify(observed[modality].float(), patch_steps=self.patch_steps, torch=torch)
                patch_present = observed_patches.any(dim=2)
                projected = self.adapters[modality](patches)
                projected = projected + self.position[:, : projected.shape[1], :] + self.modality_embeddings[modality]
                if modality in dropped_modalities:
                    projected = self.missing_tokens[modality].expand_as(projected) + self.position[:, : projected.shape[1], :] + self.modality_embeddings[modality]
                    patch_present = torch.ones_like(patch_present)
                elif modality in corruption_masks:
                    mask = corruption_masks[modality].unsqueeze(2)
                    replacement = self.mask_tokens[modality].expand_as(projected) + self.position[:, : projected.shape[1], :] + self.modality_embeddings[modality]
                    projected = torch.where(mask, replacement, projected)
                tokens.append(projected)
                padding_masks.append(~patch_present)
                slices[modality] = slice(offset, offset + projected.shape[1])
                offset += projected.shape[1]
            combined = torch.cat(tokens, dim=1)
            padding = torch.cat(padding_masks, dim=1)
            encoded = self.encoder(combined, src_key_padding_mask=padding)
            return encoded, slices, padding

        def reconstruct(
            self,
            inputs: dict[str, object],
            observed: dict[str, object],
            *,
            corruption_masks: dict[str, object],
            dropped_modalities: set[str] | None = None,
        ) -> dict[str, object]:
            encoded, slices, _ = self.encode(
                inputs,
                observed,
                corruption_masks=corruption_masks,
                dropped_modalities=dropped_modalities,
            )
            return {
                modality: self.decoders[modality](encoded[:, slices[modality], :])
                for modality in inputs
            }

        def embedding(
            self,
            inputs: dict[str, object],
            observed: dict[str, object],
            *,
            dropped_modalities: set[str] | None = None,
        ):
            encoded, _, padding = self.encode(inputs, observed, dropped_modalities=dropped_modalities)
            weights = (~padding).float().unsqueeze(2)
            return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        def forward(
            self,
            inputs: dict[str, object],
            observed: dict[str, object],
            *,
            dropped_modalities: set[str] | None = None,
        ):
            return self.occurrence_head(self.embedding(inputs, observed, dropped_modalities=dropped_modalities))

        def modality_pretraining_parameters(self, modality: str) -> list[object]:
            return [
                *self.adapters[modality].parameters(),
                *self.decoders[modality].parameters(),
                self.modality_embeddings[modality],
                self.mask_tokens[modality],
                self.missing_tokens[modality],
            ]

    construction_rng_state = torch.random.get_rng_state() if initialization_seed is not None else None
    model = MultimodalPatchTransformer()
    if initialization_seed is not None:
        initialize_named_parameters(torch, model, seed=initialization_seed)
        torch.random.set_rng_state(construction_rng_state)
    return model


def initialize_named_parameters(torch: object, model: object, *, seed: int) -> None:
    """Initialize each parameter independently so module order cannot perturb weights."""

    rng_state = torch.random.get_rng_state()
    try:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                torch.manual_seed(_parameter_seed(seed, name))
                if name.endswith("bias"):
                    torch.nn.init.zeros_(parameter)
                elif parameter.ndim == 1 and "norm" in name:
                    torch.nn.init.ones_(parameter)
                elif parameter.ndim >= 2 and not _is_embedding_parameter(name):
                    torch.nn.init.xavier_uniform_(parameter)
                else:
                    torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
    finally:
        torch.random.set_rng_state(rng_state)


def _parameter_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _is_embedding_parameter(name: str) -> bool:
    return name == "position" or any(
        name.startswith(prefix)
        for prefix in ("modality_embeddings.", "mask_tokens.", "missing_tokens.")
    )


def load_compatible_state(model: object, state: dict[str, object]) -> list[str]:
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in state.items()
        if name in current and tuple(value.shape) == tuple(current[name].shape)
    }
    model.load_state_dict(compatible, strict=False)
    return sorted(compatible)


def clone_state(model: object) -> dict[str, object]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def patchify(x, *, patch_steps: int, torch: object):
    sample_count, lookback_steps, feature_count = x.shape
    patch_count = math.ceil(lookback_steps / patch_steps)
    padded_steps = patch_count * patch_steps
    if padded_steps > lookback_steps:
        padding = torch.zeros(
            sample_count,
            padded_steps - lookback_steps,
            feature_count,
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat([x, padding], dim=1)
    return x.reshape(sample_count, patch_count, patch_steps * feature_count)
