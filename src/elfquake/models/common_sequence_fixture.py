"""Materialize common window rows as missing-aware sequence manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from elfquake.models.common_window_fixture import GROUP_PREFIXES
from elfquake.models.sequence_materializer import materialize_sequence_dataset


def materialize_common_fixture_sequences(
    *, input_csv: Path, out_root: Path, dataset_ids: list[str] | None = None
) -> dict[str, object]:
    rows, fields = _read(input_csv)
    selected_ids = dataset_ids or sorted({row.get("dataset_id", "") for row in rows if row.get("dataset_id", "")})
    numeric_fields = [field for field in fields if _numeric_column(rows, field)]
    manifests = []
    for modality, prefixes in GROUP_PREFIXES.items():
        channels = [field for field in numeric_fields if field.startswith(prefixes)]
        if not channels:
            continue
        for dataset_id in selected_ids:
            dataset_rows = sorted(
                [row for row in rows if row.get("dataset_id", "") == dataset_id],
                key=lambda row: row.get("window_start_utc", ""),
            )
            if not dataset_rows:
                continue
            safe_id = _safe(dataset_id)
            safe_modality = _safe(modality)
            out_dir = out_root / f"{safe_id}_{safe_modality}_sequence"
            input_path = out_dir / "fixture_input.csv"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["time_utc", *channels], lineterminator="\n")
                writer.writeheader()
                for row in dataset_rows:
                    output = {"time_utc": row.get("window_start_utc", "")}
                    output.update({field: row.get(field, "") for field in channels})
                    writer.writerow(output)
            manifest = materialize_sequence_dataset(
                input_csv=input_path,
                out_dir=out_dir,
                time_field="time_utc",
                entity_field=None,
                modality=modality,
                dataset_id=dataset_id,
                channel_fields=channels,
            )
            manifest.update({"fixture_source": str(input_csv), "missing_aware": 1})
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifests.append(str(out_dir / "manifest.json"))

    out_root.mkdir(parents=True, exist_ok=True)
    index_path = out_root / "manifests.txt"
    index_path.write_text("\n".join(manifests) + "\n", encoding="utf-8")
    report = {
        "schema": "elfquake.common_sequence_fixture.v1",
        "input_csv": str(input_csv),
        "dataset_ids": selected_ids,
        "modalities": sorted(GROUP_PREFIXES),
        "manifest_count": len(manifests),
        "manifest_index": str(index_path),
        "missing_modalities_are_zero_filled_and_masked": True,
    }
    (out_root / "fixture.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _numeric_column(rows: list[dict[str, str]], field: str) -> bool:
    present = [row.get(field, "") for row in rows if row.get(field, "") != ""]
    if not present:
        return False
    try:
        for value in present:
            float(value)
    except ValueError:
        return False
    return True


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
