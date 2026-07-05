"""Output helpers for sandpile simulation artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_snapshot(snapshot_dir: Path, step: int, grid: np.ndarray) -> dict[str, str]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"sandpile_step_{step:06d}.npy"
    np.save(path, grid)
    return {"step": str(step), "snapshot_file": str(path)}
