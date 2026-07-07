"""Estimate readiness and resource needs for larger model families."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def estimate_model_scale(
    *,
    input_csv: Path,
    out_path: Path,
    sequence_manifest_paths: list[Path] | None = None,
    target_field: str = "target_occurred",
    group_field: str = "dataset_id",
    lookback_steps: int = 60,
    include_missing_masks: bool = True,
) -> dict[str, object]:
    rows, fieldnames = _read_rows_and_fields(input_csv)
    labeled = [row for row in rows if row.get(target_field) in {"0", "1"}]
    positives = sum(1 for row in labeled if row.get(target_field) == "1")
    negatives = sum(1 for row in labeled if row.get(target_field) == "0")
    groups = sorted({row.get(group_field, "") for row in labeled if row.get(group_field, "")})
    numeric_features = _numeric_feature_count(labeled, fieldnames, target_field)
    sequence_summaries = [_sequence_summary(path, include_missing_masks=include_missing_masks) for path in sequence_manifest_paths or []]
    sequence_feature_count = sum(int(item["effective_feature_count"]) for item in sequence_summaries)
    bytes_per_sequence_sample = lookback_steps * sequence_feature_count * 4
    report = {
        "schema": "elfquake.model_scale_estimate.v1",
        "input": str(input_csv),
        "row_count": len(rows),
        "labeled_row_count": len(labeled),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": _rate(positives, len(labeled)),
        "group_field": group_field,
        "group_count": len(groups),
        "groups": groups,
        "numeric_tabular_feature_count": numeric_features,
        "lookback_steps": lookback_steps,
        "include_missing_masks": include_missing_masks,
        "sequence_manifest_count": len(sequence_summaries),
        "sequence_manifests": sequence_summaries,
        "sequence_feature_count": sequence_feature_count,
        "bytes_per_sequence_sample_float32": bytes_per_sequence_sample,
        "estimated_full_sequence_tensor_mb": round(bytes_per_sequence_sample * max(1, len(labeled)) / (1024 * 1024), 6),
        "gates": _gates(
            labeled_count=len(labeled),
            positives=positives,
            negatives=negatives,
            group_count=len(groups),
            sequence_feature_count=sequence_feature_count,
        ),
        "recommended_next_model": _recommended_next_model(
            labeled_count=len(labeled),
            positives=positives,
            negatives=negatives,
            group_count=len(groups),
            sequence_feature_count=sequence_feature_count,
        ),
        "cpu_guidance": _cpu_guidance(sequence_feature_count=sequence_feature_count),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _sequence_summary(path: Path, *, include_missing_masks: bool) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    channel_count = int(manifest.get("channel_count", len(manifest.get("channel_fields", []))))
    entity_count = int(manifest.get("entity_count", 1))
    time_count = int(manifest.get("time_count", 0))
    effective_feature_count = channel_count * (2 if include_missing_masks else 1)
    return {
        "path": str(path),
        "dataset_id": _dataset_id(path, manifest),
        "modality": str(manifest.get("modality", "")),
        "time_count": time_count,
        "entity_count": entity_count,
        "channel_count": channel_count,
        "effective_feature_count": effective_feature_count,
        "value_count": time_count * max(1, entity_count) * channel_count,
    }


def _gates(
    *,
    labeled_count: int,
    positives: int,
    negatives: int,
    group_count: int,
    sequence_feature_count: int,
) -> dict[str, dict[str, object]]:
    return {
        "tabular_or_gru_smoke": {
            "ready": labeled_count >= 500 and positives >= 50 and negatives >= 50,
            "minimum": ">=500 labeled rows and >=50 rows per class",
        },
        "larger_gru_or_tiny_transformer_synthetic": {
            "ready": labeled_count >= 1000 and positives >= 100 and negatives >= 100 and group_count >= 3 and sequence_feature_count > 0,
            "minimum": ">=1000 labeled rows, >=100 rows per class, >=3 groups, and sequence features",
        },
        "small_transformer_synthetic": {
            "ready": labeled_count >= 5000 and positives >= 500 and negatives >= 500 and group_count >= 5 and sequence_feature_count > 0,
            "minimum": ">=5000 labeled rows, >=500 rows per class, >=5 groups, and sequence features",
        },
        "real_transformer_training": {
            "ready": labeled_count >= 5000 and positives >= 500 and negatives >= 500,
            "minimum": ">=5000 real labeled rows and >=500 rows per class",
        },
    }


def _recommended_next_model(
    *,
    labeled_count: int,
    positives: int,
    negatives: int,
    group_count: int,
    sequence_feature_count: int,
) -> str:
    if positives == 0 or negatives == 0:
        return "do_not_train_supervised_model_until_both_classes_exist"
    if labeled_count < 500:
        return "seismic_only_and_tabular_smoke_baselines"
    if labeled_count < 1000 or group_count < 3 or sequence_feature_count == 0:
        return "tabular_mlp_or_gru_smoke_only"
    if labeled_count < 5000:
        return "larger_gru_or_tiny_patch_transformer_synthetic_only"
    return "small_patch_transformer_with_strict_ablations"


def _cpu_guidance(*, sequence_feature_count: int) -> dict[str, object]:
    if sequence_feature_count <= 0:
        return {
            "status": "no_sequence_features",
            "note": "Materialize sequence manifests before planning sequence or Transformer models.",
        }
    return {
        "status": "cpu_only",
        "tiny_transformer_start": {
            "d_model": 32,
            "layers": 2,
            "heads": 2,
            "dropout": 0.1,
            "batch_size": 32,
        },
        "small_transformer_after_data_growth": {
            "d_model": 64,
            "layers": 4,
            "heads": 4,
            "dropout": 0.1,
            "batch_size": 32,
        },
        "avoid_now": ["large attention models", "GPU-only dependencies", "claims from synthetic-only validation"],
    }


def _numeric_feature_count(rows: list[dict[str, str]], fieldnames: list[str], target_field: str) -> int:
    excluded = {target_field, "target_status", "window_id", "region_id", "dataset_id", "window_start_utc", "window_end_utc"}
    count = 0
    for field in fieldnames:
        if field in excluded:
            continue
        values = [row.get(field, "") for row in rows if row.get(field, "") != ""]
        if values and all(_is_float(value) for value in values):
            count += 1
    return count


def _read_rows_and_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _dataset_id(path: Path, manifest: dict[str, object]) -> str:
    input_csv = str(manifest.get("input_csv", ""))
    for token in [path.name, path.parent.name, input_csv]:
        if "seed40" in token:
            return "seed40"
        if "seed41" in token:
            return "seed41"
        if "seed42" in token:
            return "seed42"
    return path.parent.name


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False
