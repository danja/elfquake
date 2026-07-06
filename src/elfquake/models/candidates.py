"""Registry of replaceable model candidate families."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelCandidate:
    candidate_id: str
    name: str
    stage: str
    input_kind: str
    required_modalities: tuple[str, ...]
    optional_modalities: tuple[str, ...]
    baseline_required: bool
    external_dependencies: tuple[str, ...]
    status: str
    notes: str


MODEL_CANDIDATES: tuple[ModelCandidate, ...] = (
    ModelCandidate(
        candidate_id="historical_rate",
        name="Historical-rate baseline",
        stage="baseline",
        input_kind="region_window_table",
        required_modalities=("seismic",),
        optional_modalities=(),
        baseline_required=True,
        external_dependencies=(),
        status="planned",
        notes="Reference model using only past event frequency by region and time window.",
    ),
    ModelCandidate(
        candidate_id="regularized_logistic",
        name="Regularized logistic tabular baseline",
        stage="baseline",
        input_kind="region_window_table",
        required_modalities=("seismic",),
        optional_modalities=("vlf_image", "astronomy", "quality"),
        baseline_required=True,
        external_dependencies=(),
        status="smoke_implemented",
        notes="Dependency-light smoke trainer exists; real regularization can be swapped in later.",
    ),
    ModelCandidate(
        candidate_id="gradient_boosted_trees",
        name="Gradient-boosted trees tabular baseline",
        stage="baseline",
        input_kind="region_window_table",
        required_modalities=("seismic",),
        optional_modalities=("vlf_image", "astronomy", "quality"),
        baseline_required=True,
        external_dependencies=("scikit-learn or xgboost/lightgbm",),
        status="not_implemented",
        notes="Useful tabular nonlinearity check before deep sequence models.",
    ),
    ModelCandidate(
        candidate_id="torch_tabular_mlp",
        name="CPU PyTorch tabular MLP",
        stage="baseline",
        input_kind="region_window_table",
        required_modalities=("seismic",),
        optional_modalities=("vlf_image", "astronomy", "quality", "simulation"),
        baseline_required=True,
        external_dependencies=("torch",),
        status="implemented_synthetic_smoke",
        notes="First swappable neural backend for aligned synthetic rows; CPU-only and ablation-aware.",
    ),
    ModelCandidate(
        candidate_id="tft_tabular_sequence",
        name="Temporal Fusion Transformer style tabular sequence",
        stage="transformer",
        input_kind="regular_window_tensor",
        required_modalities=("seismic",),
        optional_modalities=("vlf_image", "astronomy", "quality"),
        baseline_required=True,
        external_dependencies=("torch",),
        status="interface_only",
        notes="Lowest-risk Transformer path once labeled fixed-window rows are sufficient.",
    ),
    ModelCandidate(
        candidate_id="patchtst_channel_independent",
        name="PatchTST-style channel-independent encoder",
        stage="transformer",
        input_kind="regular_cadence_channel_tensor",
        required_modalities=("seismic", "vlf_image"),
        optional_modalities=("astronomy", "simulation"),
        baseline_required=True,
        external_dependencies=("torch",),
        status="interface_only",
        notes="Patch tokens for dense regularly sampled channels with missing-data masks.",
    ),
    ModelCandidate(
        candidate_id="crossformer_multimodal",
        name="Crossformer-style cross-modality encoder",
        stage="transformer",
        input_kind="time_patch_by_modality_tensor",
        required_modalities=("seismic", "vlf_image"),
        optional_modalities=("astronomy", "simulation"),
        baseline_required=True,
        external_dependencies=("torch",),
        status="interface_only",
        notes="Research path for explicit time and modality attention after unimodal encoders work.",
    ),
    ModelCandidate(
        candidate_id="frequency_biased_transformer",
        name="Frequency-biased Transformer",
        stage="research",
        input_kind="time_frequency_tensor",
        required_modalities=("vlf_image",),
        optional_modalities=("piezo", "astronomy"),
        baseline_required=True,
        external_dependencies=("torch",),
        status="interface_only",
        notes="Use only after real VLF sampling metadata and frequency-derived features are stable.",
    ),
    ModelCandidate(
        candidate_id="spatio_temporal_graph_transformer",
        name="Spatio-temporal graph Transformer",
        stage="research",
        input_kind="node_time_tensor",
        required_modalities=("seismic",),
        optional_modalities=("vlf_image", "astronomy"),
        baseline_required=True,
        external_dependencies=("torch",),
        status="interface_only",
        notes="Requires a defensible Italy region/station graph before implementation.",
    ),
    ModelCandidate(
        candidate_id="transformer_hawkes",
        name="Transformer Hawkes/event-process model",
        stage="research",
        input_kind="irregular_event_stream",
        required_modalities=("seismic",),
        optional_modalities=("vlf_image", "astronomy"),
        baseline_required=True,
        external_dependencies=("torch",),
        status="interface_only",
        notes="Irregular event-stream branch; fixed-window classifiers remain primary benchmark.",
    ),
)


def list_model_candidates(*, stage: str | None = None) -> list[dict[str, object]]:
    candidates = MODEL_CANDIDATES
    if stage:
        candidates = tuple(candidate for candidate in candidates if candidate.stage == stage)
    return [_candidate_dict(candidate) for candidate in candidates]


def write_model_candidates(*, out_path: Path, stage: str | None = None) -> list[dict[str, object]]:
    rows = list_model_candidates(stage=stage)
    payload = {
        "schema": "elfquake.model_candidates.v1",
        "candidate_count": len(rows),
        "stage": stage or "",
        "candidates": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows


def _candidate_dict(candidate: ModelCandidate) -> dict[str, object]:
    row = asdict(candidate)
    row["required_modalities"] = list(candidate.required_modalities)
    row["optional_modalities"] = list(candidate.optional_modalities)
    row["external_dependencies"] = list(candidate.external_dependencies)
    return row
