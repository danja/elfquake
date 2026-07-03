"""Build aligned model rows from window and sequence tensor manifests."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from elfquake.features.common import parse_utc


BASE_INDEX_FIELDS = ["window_id", "region_id", "window_start_utc", "window_end_utc", "source_file"]
TARGET_FIELDS = ["target_event_count", "target_occurred", "target_status"]


def build_aligned_window_dataset(
    *,
    base_manifest_path: Path,
    out_path: Path,
    sequence_manifest_paths: list[Path] | None = None,
    tensor_manifest_paths: list[Path] | None = None,
    target_source_feature: str = "",
    target_horizon_rows: int = 1,
    target_threshold: float = 0.0,
    drop_unlabeled_targets: bool = False,
) -> list[dict[str, str]]:
    if target_horizon_rows < 1:
        raise ValueError("target_horizon_rows must be at least 1")

    base_manifest = _read_manifest(base_manifest_path, expected_schema="elfquake.tensor_dataset.v1")
    base_rows = _read_window_tensor_rows(base_manifest)
    sequence_sets = [_read_sequence_dataset(path) for path in sequence_manifest_paths or []]
    tensor_sets = [_read_timed_tensor_dataset(path) for path in tensor_manifest_paths or []]

    rows: list[dict[str, str]] = []
    for index, base in enumerate(base_rows):
        output = {
            field: base.get(field, "")
            for field in BASE_INDEX_FIELDS
            if field in base
        }
        output.update({field: base.get(field, "") for field in base["feature_fields"]})
        start = parse_utc(base["window_start_utc"])
        end = parse_utc(base["window_end_utc"])

        for sequence in sequence_sets:
            output.update(_aggregate_records(sequence, start=start, end=end))
        for tensor in tensor_sets:
            output.update(_aggregate_records(tensor, start=start, end=end))
        target = _target_fields(
            base_rows,
            row_index=index,
            source_feature=target_source_feature,
            horizon_rows=target_horizon_rows,
            threshold=target_threshold,
        )
        if drop_unlabeled_targets and target.get("target_occurred", "") not in {"0", "1"}:
            continue
        output.update(target)
        rows.append(output)

    fieldnames = _fieldnames(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read_manifest(path: Path, *, expected_schema: str) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema = manifest.get("schema", "")
    if schema != expected_schema:
        raise ValueError(f"expected {expected_schema}, got {schema} in {path}")
    return manifest


def _read_window_tensor_rows(manifest: dict[str, object]) -> list[dict[str, str]]:
    index_rows = _read_rows(Path(str(manifest["index_csv"])))
    value_rows = _read_rows(Path(str(manifest["values_csv"])))
    values_by_row = {row["row_index"]: row for row in value_rows}
    feature_fields = [str(field) for field in manifest.get("feature_fields", [])]
    rows = []
    for index_row in index_rows:
        value_row = values_by_row[index_row["row_index"]]
        row = dict(index_row)
        for field in feature_fields:
            row[field] = value_row.get(field, "")
        row["feature_fields"] = feature_fields
        rows.append(row)
    return rows


def _read_sequence_dataset(path: Path) -> dict[str, object]:
    manifest = _read_manifest(path, expected_schema="elfquake.sequence_dataset.v1")
    values = _read_rows(Path(str(manifest["values_csv"])))
    time_axis = _read_rows(Path(str(manifest["time_axis_csv"])))
    time_by_index = {row["time_index"]: row.get("time_utc", "") for row in time_axis}
    channels = [str(field) for field in manifest.get("channel_fields", [])]
    prefix = _field_prefix(str(manifest.get("modality", path.parent.name)))
    records = []
    for row in values:
        time_utc = time_by_index.get(row["time_index"], "")
        if not time_utc:
            continue
        records.append({"time_utc": time_utc, **{field: row.get(field, "") for field in channels}})
    return {
        "prefix": prefix,
        "channels": channels,
        "records": records,
    }


def _read_timed_tensor_dataset(path: Path) -> dict[str, object]:
    manifest = _read_manifest(path, expected_schema="elfquake.tensor_dataset.v1")
    index_rows = _read_rows(Path(str(manifest["index_csv"])))
    value_rows = _read_rows(Path(str(manifest["values_csv"])))
    values_by_row = {row["row_index"]: row for row in value_rows}
    time_field = str(manifest.get("time_field", ""))
    channels = [str(field) for field in manifest.get("feature_fields", [])]
    prefix = _tensor_prefix(manifest, path)
    records = []
    for index_row in index_rows:
        time_utc = index_row.get(time_field, "")
        if not time_utc:
            continue
        value_row = values_by_row[index_row["row_index"]]
        records.append({"time_utc": time_utc, **{field: value_row.get(field, "") for field in channels}})
    return {
        "prefix": prefix,
        "channels": channels,
        "records": records,
    }


def _aggregate_records(dataset: dict[str, object], *, start, end) -> dict[str, str]:
    prefix = str(dataset["prefix"])
    channels = [str(field) for field in dataset["channels"]]
    records = [
        record
        for record in dataset["records"]
        if start <= parse_utc(str(record["time_utc"])) < end
    ]
    output = {
        f"{prefix}_sample_count": str(len(records)),
        f"quality_missing_{prefix}": "1" if not records else "0",
    }
    for channel in channels:
        values = [_float(record.get(channel, "")) for record in records]
        values = [value for value in values if value is not None]
        field = _field_prefix(channel)
        if not values:
            output[f"{prefix}_{field}_mean"] = ""
            output[f"{prefix}_{field}_max"] = ""
            output[f"{prefix}_{field}_sum"] = ""
            continue
        output[f"{prefix}_{field}_mean"] = _fmt(sum(values) / len(values))
        output[f"{prefix}_{field}_max"] = _fmt(max(values))
        output[f"{prefix}_{field}_sum"] = _fmt(sum(values))
    return output


def _target_fields(
    base_rows: list[dict[str, str]],
    *,
    row_index: int,
    source_feature: str,
    horizon_rows: int,
    threshold: float,
) -> dict[str, str]:
    if not source_feature:
        return {}
    target_index = row_index + horizon_rows
    if target_index >= len(base_rows):
        return {
            "target_event_count": "",
            "target_occurred": "",
            "target_status": "unlabeled_no_future_window",
        }
    value = float(base_rows[target_index].get(source_feature, "0") or "0")
    return {
        "target_event_count": f"{value:g}",
        "target_occurred": "1" if value > threshold else "0",
        "target_status": "labeled",
    }


def _tensor_prefix(manifest: dict[str, object], path: Path) -> str:
    modalities = manifest.get("modalities", {})
    if isinstance(modalities, dict):
        active = [
            str(name)
            for name, summary in modalities.items()
            if isinstance(summary, dict) and int(summary.get("feature_count", 0)) > 0
        ]
        if len(active) == 1:
            return _field_prefix(active[0])
    return _field_prefix(path.parent.name)


def _fieldnames(rows: list[dict[str, str]]) -> list[str]:
    fields: list[str] = []
    preferred = [*BASE_INDEX_FIELDS, *TARGET_FIELDS]
    for field in preferred:
        if any(field in row for row in rows) and field not in fields:
            fields.append(field)
    for row in rows:
        for field in row:
            if field == "feature_fields" or field in fields:
                continue
            fields.append(field)
    return fields


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _field_prefix(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return cleaned or "dataset"


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _fmt(value: float) -> str:
    return f"{value:.9f}"
