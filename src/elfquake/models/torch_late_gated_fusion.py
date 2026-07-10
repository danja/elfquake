"""Late gated fusion around independently encoded modality branches."""

from __future__ import annotations


ANCHOR_MODALITY = "synthetic_piezo_vlf"
AUXILIARY_MODALITIES = (
    "synthetic_direct_avalanche",
    "synthetic_summary",
)


def build_late_gated_fusion_classifier(
    torch: object,
    *,
    backbone: object,
    d_model: int,
    auxiliary_modalities: tuple[str, ...] = AUXILIARY_MODALITIES,
    closed_gate_bias: float | None = None,
    initialize_from_anchor_head: bool = False,
    freeze_backbone: bool = False,
):
    class LateGatedFusionClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.auxiliary_modalities = auxiliary_modalities
            self.auxiliary_projections = torch.nn.ModuleDict({
                modality: torch.nn.Linear(d_model, d_model)
                for modality in self.auxiliary_modalities
            })
            self.gates = torch.nn.ModuleDict({
                modality: torch.nn.Sequential(
                    torch.nn.Linear(d_model * 2, d_model),
                    torch.nn.GELU(),
                    torch.nn.Linear(d_model, 1),
                )
                for modality in self.auxiliary_modalities
            })
            self.fusion_norm = torch.nn.LayerNorm(d_model)
            self.occurrence_head = torch.nn.Linear(d_model, 1)
            self.freeze_backbone_for_fusion = freeze_backbone
            if closed_gate_bias is not None:
                for gate in self.gates.values():
                    torch.nn.init.zeros_(gate[-1].weight)
                    torch.nn.init.constant_(gate[-1].bias, closed_gate_bias)
            if initialize_from_anchor_head:
                self.occurrence_head.load_state_dict(self.backbone.occurrence_head.state_dict())

        def train(self, mode: bool = True):
            super().train(mode)
            if self.freeze_backbone_for_fusion:
                self.backbone.eval()
            return self

        def fusion_parameters(self) -> list[object]:
            return [
                *self.auxiliary_projections.parameters(),
                *self.gates.parameters(),
                *self.fusion_norm.parameters(),
                *self.occurrence_head.parameters(),
            ]

        def embedding(
            self,
            inputs: dict[str, object],
            observed: dict[str, object],
            *,
            dropped_modalities: set[str] | None = None,
        ):
            fused, _ = self.embedding_with_gates(
                inputs,
                observed,
                dropped_modalities=dropped_modalities,
            )
            return fused

        def embedding_with_gates(
            self,
            inputs: dict[str, object],
            observed: dict[str, object],
            *,
            dropped_modalities: set[str] | None = None,
        ):
            dropped = dropped_modalities or set()
            context = torch.no_grad() if self.freeze_backbone_for_fusion else _NullContext()
            with context:
                anchor = self.backbone.embedding(
                    {ANCHOR_MODALITY: inputs[ANCHOR_MODALITY]},
                    {ANCHOR_MODALITY: observed[ANCHOR_MODALITY]},
                    dropped_modalities={ANCHOR_MODALITY} if ANCHOR_MODALITY in dropped else set(),
                )
            fused = anchor
            gate_values = {}
            for modality in self.auxiliary_modalities:
                if modality in dropped:
                    gate_values[modality] = torch.zeros(anchor.shape[0], 1, dtype=anchor.dtype, device=anchor.device)
                    continue
                context = torch.no_grad() if self.freeze_backbone_for_fusion else _NullContext()
                with context:
                    auxiliary = self.backbone.embedding(
                        {modality: inputs[modality]},
                        {modality: observed[modality]},
                    )
                gate = torch.sigmoid(self.gates[modality](torch.cat([anchor, auxiliary], dim=1)))
                fused = fused + gate * self.auxiliary_projections[modality](auxiliary)
                gate_values[modality] = gate
            return self.fusion_norm(fused), gate_values

        def forward(
            self,
            inputs: dict[str, object],
            observed: dict[str, object],
            *,
            dropped_modalities: set[str] | None = None,
        ):
            return self.occurrence_head(
                self.embedding(inputs, observed, dropped_modalities=dropped_modalities)
            )

    return LateGatedFusionClassifier()


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False
