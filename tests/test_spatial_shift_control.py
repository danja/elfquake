"""Circular-shift within-cell null control (next-actions item 105(d))."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from elfquake.models.spatial_permutation import (
    circularly_shift_spatial_targets,
    permute_spatial_target_vectors,
)

FIELDS = [
    "window_start_utc",
    "target_cell_id",
    "target_occurred",
    "target_event_count",
    "feature_a",
]


def _write_table(path: Path, series: dict[str, list[str]]) -> None:
    """One row per (anchor, cell). `series` maps a cell to its label sequence."""
    length = len(next(iter(series.values())))
    rows = []
    for index in range(length):
        for cell, labels in series.items():
            rows.append({
                "window_start_utc": f"2026-07-01T{index // 60:02d}:{index % 60:02d}:00Z",
                "target_cell_id": cell,
                "target_occurred": labels[index],
                "target_event_count": labels[index],
                "feature_a": str(index),
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _series(path: Path) -> dict[str, list[str]]:
    rows = [
        row
        for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
        if row["target_occurred"] in {"0", "1"}
    ]
    out: dict[str, list[str]] = {}
    for row in sorted(rows, key=lambda r: (r["target_cell_id"], r["window_start_utc"])):
        out.setdefault(row["target_cell_id"], []).append(row["target_occurred"])
    return out


def _transitions(labels: list[str]) -> int:
    return sum(1 for a, b in zip(labels, labels[1:]) if a != b)


class CircularShiftTests(unittest.TestCase):
    def _run(self, series, seed=7, **kwargs):
        directory = tempfile.mkdtemp()
        source = Path(directory) / "in.csv"
        target = Path(directory) / "out.csv"
        _write_table(source, series)
        report = circularly_shift_spatial_targets(
            input_csv=source, out_path=target, seed=seed, **kwargs
        )
        return report, _series(source), _series(target)

    def test_length_and_positive_count_are_preserved_exactly(self) -> None:
        series = {
            "a": ["0"] * 20 + ["1"] * 10 + ["0"] * 30,
            "b": ["1"] * 5 + ["0"] * 55,
        }
        _, before, after = self._run(series)
        for cell in before:
            self.assertEqual(len(after[cell]), len(before[cell]))
            self.assertEqual(after[cell].count("1"), before[cell].count("1"))

    def test_run_structure_survives_apart_from_the_wrap_seam(self) -> None:
        # This is the property the timestamp shuffle destroys: one contiguous
        # block must stay one contiguous block, so the control is scored on the
        # same amount of evidence as the run it nulls.
        series = {"a": ["0"] * 20 + ["1"] * 10 + ["0"] * 30}
        _, before, after = self._run(series)
        self.assertLessEqual(
            abs(_transitions(after["a"]) - _transitions(before["a"])), 2
        )

    def test_shuffling_manufactures_transitions_where_shifting_does_not(self) -> None:
        # The reason this control exists. Same input, both controls; the shuffle
        # inflates the evidence count and the shift does not.
        series = {"a": ["0"] * 30 + ["1"] * 30}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "in.csv"
            shifted = Path(directory) / "shifted.csv"
            shuffled = Path(directory) / "shuffled.csv"
            _write_table(source, series)
            circularly_shift_spatial_targets(
                input_csv=source, out_path=shifted, seed=3
            )
            permute_spatial_target_vectors(input_csv=source, out_path=shuffled, seed=3)
            shift_transitions = _transitions(_series(shifted)["a"])
            shuffle_transitions = _transitions(_series(shuffled)["a"])

        self.assertLessEqual(shift_transitions, 2)
        self.assertGreater(shuffle_transitions, shift_transitions * 3)

    def test_labels_actually_move(self) -> None:
        series = {"a": ["0"] * 20 + ["1"] * 10 + ["0"] * 30}
        report, before, after = self._run(series)
        self.assertNotEqual(before["a"], after["a"])
        self.assertGreater(report["shift_anchors"], 0)

    def test_unlabeled_rows_keep_their_empty_labels(self) -> None:
        series = {"a": ["0"] * 30 + [""] * 10 + ["1"] * 20}
        _, before, after = self._run(series)
        # The empty rows are excluded from both series, so the labeled length is
        # unchanged and no empty label has been overwritten with a number.
        self.assertEqual(len(after["a"]), 50)
        self.assertEqual(after["a"].count("1"), 20)

    def test_report_records_the_transition_counts_it_preserved(self) -> None:
        series = {"a": ["0"] * 20 + ["1"] * 10 + ["0"] * 30, "b": ["0"] * 60}
        report, _, _ = self._run(series)
        self.assertEqual(report["cell_count"], 2)
        self.assertEqual(report["labeled_time_count"], 60)
        self.assertLessEqual(
            abs(int(report["transitions_after"]) - int(report["transitions_before"])), 2
        )

    def test_two_anchors_are_required(self) -> None:
        with self.assertRaises(ValueError):
            self._run({"a": ["1"]})

    def test_missing_cell_field_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._run({"a": ["0", "1"] * 10}, cell_field="not_a_column")


if __name__ == "__main__":
    unittest.main()
