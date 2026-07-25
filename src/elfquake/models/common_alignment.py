"""Audit whether common fixture sources are genuinely co-observed."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

from elfquake.models.common_window_fixture import GROUP_PREFIXES


def audit_common_window_alignment(*, input_csv: Path, out_path: Path) -> dict[str, object]:
    rows = _read(input_csv)
    datasets = sorted({row.get("dataset_id", "") for row in rows if row.get("dataset_id", "")})
    dataset_rows = {dataset: [row for row in rows if row.get("dataset_id") == dataset] for dataset in datasets}
    modality_presence = {
        dataset: {
            group: sum(_present(row, prefixes) for row in source_rows)
            for group, prefixes in GROUP_PREFIXES.items()
        }
        for dataset, source_rows in dataset_rows.items()
    }
    coverage = {
        dataset: _coverage([row.get("window_start_utc", "") for row in source_rows])
        for dataset, source_rows in dataset_rows.items()
    }
    observed_times = {
        dataset: {
            group: sorted(
                row.get("window_start_utc", "")
                for row in source_rows
                if _present(row, GROUP_PREFIXES[group]) and row.get("window_start_utc", "")
            )
            for group in GROUP_PREFIXES
        }
        for dataset, source_rows in dataset_rows.items()
    }
    pairwise = []
    for left, right in combinations(datasets, 2):
        left_times = set(_flatten(observed_times[left]))
        right_times = set(_flatten(observed_times[right]))
        pairwise.append({
            "left_dataset": left,
            "right_dataset": right,
            "interval_overlap": _interval_overlap(coverage[left], coverage[right]),
            "exact_observed_window_matches": len(left_times & right_times),
        })
    report = {
        "schema": "elfquake.common_window_alignment.v1",
        "input_csv": str(input_csv),
        "row_count": len(rows),
        "dataset_count": len(datasets),
        "dataset_row_counts": {dataset: len(dataset_rows[dataset]) for dataset in datasets},
        "dataset_coverage": coverage,
        "modality_presence_rows": modality_presence,
        "pairwise_dataset_alignment": pairwise,
        "eligible_same_row_groups": _eligible_groups(rows),
        "gates": {
            "coobserved_seismic_italy_vlf_astronomy": _count_rows(
                rows, ("seismic", "italy_vlf", "astronomy")
            ),
            "coobserved_seismic_japan_vlf": _count_rows(rows, ("seismic", "japan_vlf")),
            "coobserved_synthetic_vlf_and_direct": _count_rows(
                rows, ("synthetic_piezo_vlf", "synthetic_direct_avalanche")
            ),
        },
        "interpretation": [
            "Interval overlap is not evidence of co-observation; exact window matches and same-row modality presence are counted separately.",
            "Rows with missing modality values do not qualify for a co-observed gate.",
            "The current fixture is suitable for interface and masked-ablation tests, not cross-domain scientific skill claims.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _present(row: dict[str, str], prefixes: tuple[str, ...]) -> bool:
    for field in row:
        if not field.startswith(prefixes):
            continue
        value = row.get(field, "")
        if value in {"", None}:
            continue
        if field.endswith(("_row_count", "_capture_count", "_coverage_seconds", "_total_bytes")):
            try:
                if float(value) <= 0:
                    continue
            except ValueError:
                pass
        return True
    return False


def _count_rows(rows: list[dict[str, str]], groups: tuple[str, ...]) -> int:
    return sum(all(_present(row, GROUP_PREFIXES[group]) for group in groups) for row in rows)


def _eligible_groups(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        "+".join(groups): _count_rows(rows, groups)
        for size in range(2, 4)
        for groups in combinations(GROUP_PREFIXES, size)
        if _count_rows(rows, groups)
    }


def _coverage(values: list[str]) -> dict[str, object]:
    values = sorted(value for value in values if value)
    return {"count": len(values), "start": values[0], "end": values[-1]} if values else {"count": 0}


def _interval_overlap(left: dict[str, object], right: dict[str, object]) -> bool:
    if not left.get("start") or not right.get("start"):
        return False
    return max(str(left["start"]), str(right["start"])) <= min(str(left["end"]), str(right["end"]))


def _flatten(values: dict[str, list[str]]) -> list[str]:
    output = []
    for times in values.values():
        output.extend(times)
    return output
