"""Build a provenance-preserving, missing-aware common window fixture."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from elfquake.models.channel_gate import (
    audit_channels,
    mask_fields,
    numeric_channels,
    raise_for_defects,
)


GROUP_PREFIXES = {
    "seismic": ("seismic_",),
    "astronomy": ("astro_", "astronomy_", "moon_", "solar_", "kp_", "ap_", "f107_"),
    "italy_vlf": ("vlf_",),
    "japan_vlf": ("japan_",),
    "synthetic_piezo_vlf": ("synthetic_piezo_vlf_",),
    "synthetic_direct_avalanche": ("synthetic_direct_avalanche_",),
    "synthetic_summary": ("synthetic_summary_",),
    "synthetic_seismic": ("synthetic_seismic_",),
}


def build_common_window_fixture(
    *, input_csvs: list[Path], out_path: Path, report_path: Path,
    dataset_ids: list[str] | None = None, train_fraction: float = 0.8,
    strict_channels: bool = True, allow_constant_channels: list[str] | None = None,
) -> dict[str, object]:
    if not input_csvs:
        raise ValueError("at least one input CSV is required")
    if dataset_ids is not None and len(dataset_ids) != len(input_csvs):
        raise ValueError("dataset_ids length must match input_csvs")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    source_rows = []
    fieldnames = ["dataset_id", "fixture_source", "model_split"]
    for index, path in enumerate(input_csvs):
        rows, fields = _read(path)
        fallback_id = dataset_ids[index] if dataset_ids else path.stem
        for row in rows:
            output = dict(row)
            output["dataset_id"] = output.get("dataset_id", "") or fallback_id
            output["fixture_source"] = str(path)
            source_rows.append(output)
        for field in fields:
            if field not in fieldnames:
                fieldnames.append(field)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        grouped[row["dataset_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row.get("window_start_utc", ""))
        # Rows still awaiting label maturity (e.g. this week's not-yet-matured
        # prospective windows) must not count toward the chronological split:
        # letting a growing pending-future tail dominate the row count pushes
        # the train/test boundary past every mature row, leaving test empty.
        usable = [row for row in rows if row.get("target_status", "") in ("", "labeled")]
        split_at = max(1, min(len(usable) - 1, int(len(usable) * train_fraction))) if len(usable) > 1 else len(usable)
        usable_index = 0
        for row in rows:
            if row.get("target_status", "") not in ("", "labeled"):
                row["model_split"] = "test"
                continue
            row["model_split"] = "train" if usable_index < split_at else "test"
            usable_index += 1

    for group in GROUP_PREFIXES:
        fieldnames.append(f"quality_fixture_{group}_present")
        fieldnames.append(f"quality_fixture_{group}_missing")
    rows = [row for group_rows in grouped.values() for row in group_rows]
    numeric_fields = [field for field in fieldnames if field not in {"dataset_id", "fixture_source", "model_split"}]
    for row in rows:
        for group, prefixes in GROUP_PREFIXES.items():
            present = any(
                _has_value(row.get(field, ""), field)
                for field in numeric_fields
                if field.startswith(prefixes)
            )
            row[f"quality_fixture_{group}_present"] = "1" if present else "0"
            row[f"quality_fixture_{group}_missing"] = "0" if present else "1"
    fieldnames.extend(field for field in fieldnames if field not in fieldnames)
    rows.sort(key=lambda row: (row["dataset_id"], row.get("window_start_utc", "")))

    # A dataset that never carried a modality has no opinion column for it, so
    # the unioned row reads blank. Blank is not "present": it means the source
    # asserted nothing, which for a missing-flag is the same as missing. Left
    # unfilled, a channel blank on those rows looks silently imputed.
    for row in rows:
        for field in fieldnames:
            if field.startswith("quality_missing_") and not row.get(field):
                row[field] = "1"

    # Gate the assembled channels before anything writes them. A constant or
    # silently imputed channel reaches the model looking exactly like a real
    # one, so the check has to happen here rather than at training time.
    all_prefixes = tuple(prefix for prefixes in GROUP_PREFIXES.values() for prefix in prefixes)
    channel_defects = audit_channels(
        rows,
        channels=numeric_channels(rows, fieldnames=fieldnames, prefixes=all_prefixes),
        mask_fields=mask_fields(fieldnames),
        allow_constant=frozenset(allow_constant_channels or ()),
    )
    if strict_channels:
        raise_for_defects(channel_defects)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema": "elfquake.common_window_fixture.v1",
        "input_csvs": [str(path) for path in input_csvs],
        "output_csv": str(out_path),
        "row_count": len(rows),
        "dataset_count": len(grouped),
        "dataset_row_counts": {key: len(value) for key, value in sorted(grouped.items())},
        "split_counts": dict(Counter(row["model_split"] for row in rows)),
        "target_counts": dict(Counter(row.get("target_occurred", "") for row in rows)),
        "modality_presence": {
            group: sum(row[f"quality_fixture_{group}_present"] == "1" for row in rows)
            for group in GROUP_PREFIXES
        },
        "research_use_only_inputs": [str(path) for path in input_csvs if _research_only(path)],
        "channel_gate": {
            "strict": strict_channels,
            "allowed_constant_channels": sorted(allow_constant_channels or ()),
            "defects": [
                {"channel": defect.channel, "defect": defect.defect, "detail": defect.detail}
                for defect in channel_defects
            ],
        },
        "notes": [
            "Rows are unioned by dataset; unrelated time ranges are not cross-joined.",
            "model_split is chronological within dataset_id.",
            "Modality presence flags are explicit and do not impute missing sources.",
            "Channel gate rejects constant, empty, and silently imputed channels.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _has_value(value: str, field: str) -> bool:
    if value in {"", None}:
        return False
    if not field.endswith(("_row_count", "_capture_count", "_coverage_seconds", "_total_bytes")):
        return True
    return not _zero_only_metadata(value)


def _zero_only_metadata(value: str) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _research_only(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8") as handle:
        return "research_use_only" in (next(csv.DictReader(handle), {}) or {})
