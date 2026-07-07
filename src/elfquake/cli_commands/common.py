"""Shared CLI output helpers."""

from __future__ import annotations

from pathlib import Path

from elfquake.storage import StoredCapture


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def resolve_image_paths(
    *,
    image_paths: list[Path],
    image_roots: list[Path],
    filename_prefixes: list[str],
) -> list[Path]:
    paths = list(image_paths)
    prefixes = tuple(prefix for prefix in filename_prefixes if prefix)
    for root in image_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if prefixes and not path.name.startswith(prefixes):
                continue
            paths.append(path)
    return paths


def print_stored_captures(stored: list[StoredCapture]) -> int:
    print(f"captures: {len(stored)}")
    for capture in stored:
        status = "skipped" if capture.skipped_existing else "stored"
        print(f"{status}: {capture.payload_path}")
        print(f"metadata: {capture.metadata_path}")
    return 0


def print_holdout_report(report: dict[str, object], out_path: Path) -> None:
    print(f"status: {report['status']}")
    print(f"rows: {report['row_count']}")
    print(f"labeled rows: {report['labeled_row_count']}")
    if "train_row_count" in report:
        print(f"train rows: {report['train_row_count']}")
        print(f"test rows: {report['test_row_count']}")
    print(f"output: {out_path}")
