"""Shared CLI helpers."""

from __future__ import annotations

from pathlib import Path


def print_stored_captures(stored) -> int:
    for capture in stored:
        status = "skipped" if capture.skipped_existing else "stored"
        print(f"{status}: {capture.payload_path}")
        print(f"metadata: {capture.metadata_path}")
    return 0


def resolve_image_paths(
    *,
    image_paths: list[Path],
    image_roots: list[Path],
    filename_prefixes: list[str],
) -> list[Path]:
    resolved = list(image_paths)
    for root in image_roots:
        resolved.extend(sorted(root.glob("**/*.jpg")))
        resolved.extend(sorted(root.glob("**/*.jpeg")))
    if filename_prefixes:
        resolved = [
            path for path in resolved if any(path.name.startswith(prefix) for prefix in filename_prefixes)
        ]
    unique = sorted(dict.fromkeys(resolved))
    if not unique:
        raise ValueError("at least one --image or matching --image-root file is required")
    return unique
