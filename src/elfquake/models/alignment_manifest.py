"""Model-run alignment manifests across materialized datasets."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


TIMESTAMP_PATTERN = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<hour>\d{2})[-:](?P<minute>\d{2})[-:](?P<second>\d{2})Z"
)


def build_alignment_manifest(
    *,
    manifest_paths: list[Path],
    out_path: Path,
    run_id: str,
) -> dict[str, object]:
    if not manifest_paths:
        raise ValueError("at least one manifest path is required")

    datasets = [_dataset_summary(path) for path in manifest_paths]
    report: dict[str, object] = {
        "schema": "elfquake.alignment_manifest.v1",
        "run_id": run_id,
        "dataset_count": len(datasets),
        "datasets": datasets,
        "ablation_groups": _ablation_groups(datasets),
        "alignment_notes": _alignment_notes(datasets),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _dataset_summary(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema = str(manifest.get("schema", ""))
    if schema == "elfquake.tensor_dataset.v1":
        return _tensor_summary(path, manifest)
    if schema == "elfquake.sequence_dataset.v1":
        return _sequence_summary(path, manifest)
    raise ValueError(f"unsupported manifest schema in {path}: {schema}")


def _tensor_summary(path: Path, manifest: dict[str, object]) -> dict[str, object]:
    modalities = _tensor_modalities(manifest)
    index_csv = Path(str(manifest.get("index_csv", "")))
    time_field = str(manifest.get("time_field", ""))
    time_coverage = _index_time_coverage(index_csv, time_field) if index_csv else {}
    return {
        "name": path.parent.name,
        "kind": "window_tensor",
        "schema": manifest.get("schema", ""),
        "manifest_path": str(path),
        "layout": manifest.get("layout", ""),
        "row_count": manifest.get("row_count", 0),
        "feature_count": manifest.get("feature_count", 0),
        "modalities": modalities,
        "values_csv": manifest.get("values_csv", ""),
        "masks_csv": manifest.get("masks_csv", ""),
        "index_csv": manifest.get("index_csv", ""),
        "supports_missing_masks": bool(manifest.get("masks_csv") and manifest.get("mask_fields")),
        "time_coverage": time_coverage,
    }


def _sequence_summary(path: Path, manifest: dict[str, object]) -> dict[str, object]:
    time_axis_csv = Path(str(manifest.get("time_axis_csv", "")))
    entity_axis_csv = Path(str(manifest.get("entity_axis_csv", "")))
    return {
        "name": path.parent.name,
        "kind": "sequence_tensor",
        "schema": manifest.get("schema", ""),
        "manifest_path": str(path),
        "layout": manifest.get("layout", ""),
        "row_count": manifest.get("row_count", 0),
        "time_count": manifest.get("time_count", 0),
        "entity_count": manifest.get("entity_count", 0),
        "channel_count": manifest.get("channel_count", 0),
        "modalities": [str(manifest.get("modality", ""))],
        "values_csv": manifest.get("values_csv", ""),
        "masks_csv": manifest.get("masks_csv", ""),
        "index_csv": manifest.get("index_csv", ""),
        "time_axis_csv": manifest.get("time_axis_csv", ""),
        "entity_axis_csv": manifest.get("entity_axis_csv", ""),
        "supports_missing_masks": bool(manifest.get("masks_csv") and manifest.get("mask_fields")),
        "time_coverage": _axis_coverage(time_axis_csv, str(manifest.get("time_field", "time"))) if time_axis_csv else {},
        "entity_coverage": _axis_coverage(entity_axis_csv, str(manifest.get("entity_field", "entity"))) if entity_axis_csv else {},
    }


def _tensor_modalities(manifest: dict[str, object]) -> list[str]:
    modalities = manifest.get("modalities", {})
    if not isinstance(modalities, dict):
        return []
    names = []
    for name, summary in modalities.items():
        if isinstance(summary, dict) and int(summary.get("feature_count", 0)) > 0:
            names.append(str(name))
    return names


def _index_time_coverage(index_csv: Path, time_field: str) -> dict[str, object]:
    rows = _read_rows(index_csv)
    if not rows:
        return {"count": 0}
    if time_field and time_field in rows[0]:
        values = []
        for row in rows:
            value = row.get(time_field, "")
            if not value:
                continue
            extracted = _timestamps_from_value(value)
            values.extend(extracted or [value])
    else:
        values = []
        for row in rows:
            values.extend(_timestamps_from_row(row))
    return _value_coverage(values)


def _axis_coverage(axis_csv: Path, value_field: str) -> dict[str, object]:
    rows = _read_rows(axis_csv)
    if not rows:
        return {"count": 0}
    field = value_field if value_field in rows[0] else next((name for name in rows[0] if not name.endswith("_index")), "")
    values = [row.get(field, "") for row in rows if row.get(field, "")]
    return _value_coverage(values)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _timestamps_from_row(row: dict[str, str]) -> list[str]:
    values = []
    for value in row.values():
        values.extend(_timestamps_from_value(value))
    return values


def _timestamps_from_value(value: str) -> list[str]:
    values = []
    for match in TIMESTAMP_PATTERN.finditer(value):
        values.append(
            f"{match.group('date')}T{match.group('hour')}:{match.group('minute')}:{match.group('second')}Z"
        )
    return values


def _value_coverage(values: list[str]) -> dict[str, object]:
    if not values:
        return {"count": 0}
    sortable = sorted(values, key=_sort_key)
    return {
        "count": len(values),
        "start": sortable[0],
        "end": sortable[-1],
    }


def _sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _ablation_groups(datasets: list[dict[str, object]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for dataset in datasets:
        for modality in dataset.get("modalities", []):
            if not modality:
                continue
            groups.setdefault(str(modality), []).append(str(dataset["name"]))
    return groups


def _alignment_notes(datasets: list[dict[str, object]]) -> list[str]:
    notes = []
    if any(dataset["kind"] == "sequence_tensor" for dataset in datasets):
        notes.append("sequence tensors need an explicit time-scale mapping before joining to UTC event windows")
    if any("vlf_image" in dataset.get("modalities", []) for dataset in datasets):
        notes.append("VLF image tensors use capture-time inference from source filenames unless an explicit time field is added")
    if "synthetic_piezo_vlf" in _ablation_groups(datasets) and "synthetic_direct_avalanche" in _ablation_groups(datasets):
        notes.append("synthetic piezo/VLF and direct avalanche/seismic channels must remain separate ablation groups")
    return notes
