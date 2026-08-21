"""Null controls that preserve spatial label patterns while removing time order."""

from __future__ import annotations

import csv
import random
from pathlib import Path


TARGET_VECTOR_FIELDS = (
    "target_event_count",
    "target_occurred",
    "target_magnitude_min",
    "target_start_utc",
    "target_end_utc",
    "target_status",
)


def permute_spatial_target_vectors(
    *, input_csv: Path, out_path: Path, seed: int = 42, time_field: str = "window_start_utc"
) -> dict[str, int | str]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames or time_field not in fieldnames:
        raise ValueError(f"missing time field: {time_field}")

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get(time_field, ""), []).append(row)
    labeled_times = [
        time
        for time, group in groups.items()
        if group and all(row.get("target_occurred", "") in {"0", "1"} for row in group)
    ]
    vectors = [
        [[row.get(field, "") for field in TARGET_VECTOR_FIELDS] for row in groups[time]]
        for time in labeled_times
    ]
    random.Random(seed).shuffle(vectors)
    for time, vector in zip(labeled_times, vectors):
        for row, source in zip(groups[time], vector):
            for field, value in zip(TARGET_VECTOR_FIELDS, source):
                if field in row:
                    row[field] = value

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {"row_count": len(rows), "labeled_time_count": len(labeled_times), "seed": seed, "output": str(out_path)}


def circularly_shift_spatial_targets(
    *,
    input_csv: Path,
    out_path: Path,
    seed: int = 42,
    time_field: str = "window_start_utc",
    cell_field: str = "target_cell_id",
    min_shift_fraction: float = 0.1,
) -> dict[str, object]:
    """Shift the label matrix in time, keeping each cell's run structure intact.

    `permute_spatial_target_vectors` shuffles whole time slices, which destroys
    the temporal autocorrelation along with the signal. Because consecutive
    target windows overlap by more than 99%, that autocorrelation is most of
    what the labels are: shuffling turns a handful of long runs into many short
    ones and *manufactures* label transitions. Measured on the item-104 tables
    the shuffled controls carried 94-342 held-out transitions against the real
    runs' 1 and 19, so they were scored on a task with up to a hundred times the
    evidence and could not null anything (see `docs/target-design.md`).

    A circular shift is the null that shuffling was meant to be. The whole
    labeled matrix moves by one offset, so within every cell the sequence
    length, the positive count, and the run-length structure are preserved
    exactly -- the transition count changes only where the wrap seam falls --
    and the spatial pattern at each anchor stays intact. What it destroys is the
    alignment between features and labels, which is the only thing a temporal
    signal could live in.

    Unlabeled rows keep their empty labels and do not enter the shift; the
    evaluator drops them anyway, and moving them would change which anchors are
    scoreable and reintroduce the asymmetry this control exists to remove.
    """
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for field in (time_field, cell_field):
        if not fieldnames or field not in fieldnames:
            raise ValueError(f"missing field: {field}")

    labeled_times = sorted({
        row[time_field]
        for row in rows
        if row.get("target_occurred", "") in {"0", "1"}
    })
    if len(labeled_times) < 2:
        raise ValueError("need at least two labeled anchors to shift")

    # Keep the shift away from both ends so the control is not a near-copy of
    # the real ordering, and never let it be a whole-length no-op.
    low = max(1, int(len(labeled_times) * min_shift_fraction))
    high = len(labeled_times) - low
    if high <= low:
        low, high = 1, len(labeled_times) - 1
    shift = random.Random(seed).randint(low, high)

    shifted_time = {
        time: labeled_times[(index + shift) % len(labeled_times)]
        for index, time in enumerate(labeled_times)
    }
    source: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        if row.get("target_occurred", "") in {"0", "1"}:
            source[(row[time_field], row[cell_field])] = [
                row.get(field, "") for field in TARGET_VECTOR_FIELDS
            ]

    before = _transitions_by_cell(rows, time_field=time_field, cell_field=cell_field)
    for row in rows:
        if row.get("target_occurred", "") not in {"0", "1"}:
            continue
        donor = source.get((shifted_time[row[time_field]], row[cell_field]))
        if donor is None:
            continue
        for field, value in zip(TARGET_VECTOR_FIELDS, donor):
            if field in row:
                row[field] = value
    after = _transitions_by_cell(rows, time_field=time_field, cell_field=cell_field)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "row_count": len(rows),
        "labeled_time_count": len(labeled_times),
        "cell_count": len(before),
        "shift_anchors": shift,
        "seed": seed,
        "transitions_before": sum(before.values()),
        "transitions_after": sum(after.values()),
        "output": str(out_path),
    }


def _transitions_by_cell(
    rows: list[dict[str, str]], *, time_field: str, cell_field: str
) -> dict[str, int]:
    """Label changes between consecutive labeled anchors, per cell."""
    series: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        label = row.get("target_occurred", "")
        if label in {"0", "1"}:
            series.setdefault(row[cell_field], []).append((row[time_field], label))
    counts = {}
    for cell, entries in series.items():
        ordered = [label for _, label in sorted(entries)]
        counts[cell] = sum(
            1 for previous, current in zip(ordered, ordered[1:]) if previous != current
        )
    return counts
