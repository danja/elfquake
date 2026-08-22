"""Cover the shared freshness guard used by the `all_available` scripts.

Item 89 found the same defect twice and item 104 found it a third time: a
script reads one input a refresh rebuilds and another nothing rebuilds, and
the run looks current. `scripts/lib/staleness.sh` is the shared check; these
tests pin its three behaviours so a later edit cannot quietly make it pass
everything.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "scripts" / "lib" / "staleness.sh"
# The scripts run from the repo root, where the lib's relative default for
# PYTHON_BIN resolves; these tests run from a tmp_path, so name it outright.
PYTHON_BIN = str(REPO / ".venv" / "bin" / "python")


def _run(script: str, cwd: Path, env: dict[str, str] | None = None):
    merged = dict(os.environ)
    merged["PYTHON_BIN"] = PYTHON_BIN
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash", "-c", f'. "{LIB}"\n{script}'],
        cwd=cwd,
        env=merged,
        capture_output=True,
        text=True,
    )


def _write(path: Path, text: str, mtime: float) -> Path:
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_fresh_inputs_pass_silently(tmp_path: Path) -> None:
    reference = _write(tmp_path / "catalog.csv", "a", 1_000.0)
    fresh = _write(tmp_path / "scores.csv", "b", 2_000.0)
    result = _run(f'require_fresh_inputs "{reference}" "{fresh}"', tmp_path)
    assert result.returncode == 0
    assert result.stderr == ""


def test_stale_input_warns_but_does_not_stop_the_run(tmp_path: Path) -> None:
    reference = _write(tmp_path / "catalog.csv", "a", 2_000.0)
    stale = _write(tmp_path / "scores.csv", "b", 1_000.0)
    result = _run(f'require_fresh_inputs "{reference}" "{stale}"', tmp_path)
    # Warning by default: an audit run should still produce its report, with
    # the caveat on stderr where the operator sees it.
    assert result.returncode == 0
    assert "staleness guard: 1 input(s) older" in result.stderr
    assert "scores.csv" in result.stderr


def test_stale_inputs_fail_when_asked(tmp_path: Path) -> None:
    reference = _write(tmp_path / "catalog.csv", "a", 2_000.0)
    _write(tmp_path / "scores.csv", "b", 1_000.0)
    _write(tmp_path / "windows.csv", "c", 1_500.0)
    result = _run(
        f'require_fresh_inputs "{tmp_path}/catalog.csv" "{tmp_path}/scores.csv" "{tmp_path}/windows.csv"',
        tmp_path,
        env={"STALE_INPUTS": "fail"},
    )
    assert result.returncode == 3
    assert "2 input(s) older" in result.stderr
    assert str(reference.name) in result.stderr


def test_missing_input_is_not_reported_as_stale(tmp_path: Path) -> None:
    # A path that does not exist is a different failure and the command that
    # needs it reports it in its own terms.
    reference = _write(tmp_path / "catalog.csv", "a", 2_000.0)
    result = _run(
        f'require_fresh_inputs "{reference}" "{tmp_path}/absent.csv"',
        tmp_path,
        env={"STALE_INPUTS": "fail"},
    )
    assert result.returncode == 0
    assert result.stderr == ""


def test_missing_reference_skips_the_check(tmp_path: Path) -> None:
    _write(tmp_path / "scores.csv", "b", 1_000.0)
    result = _run(
        f'require_fresh_inputs "{tmp_path}/absent.csv" "{tmp_path}/scores.csv"',
        tmp_path,
        env={"STALE_INPUTS": "fail"},
    )
    assert result.returncode == 0
    assert "skipping check" in result.stderr


def test_catalog_coverage_end_reads_the_freshness_report(tmp_path: Path) -> None:
    report = tmp_path / "catalog_freshness.json"
    report.write_text(
        json.dumps({"schema": "elfquake.catalog_freshness.v1", "coverage_end_utc": "2026-08-22T08:00:00Z"}),
        encoding="utf-8",
    )
    result = _run(f'catalog_coverage_end "{report}" "FALLBACK"', tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == "2026-08-22T08:00:00Z"


def test_catalog_coverage_end_falls_back_without_a_report(tmp_path: Path) -> None:
    result = _run(
        f'catalog_coverage_end "{tmp_path}/absent.json" "2026-07-08T00:00:00Z"',
        tmp_path,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "2026-07-08T00:00:00Z"


def test_catalog_coverage_end_falls_back_on_an_empty_coverage_field(tmp_path: Path) -> None:
    # A fetch that failed before any event was ever ingested leaves the field
    # empty; that must not print an empty as-of date into a CLI argument.
    report = tmp_path / "catalog_freshness.json"
    report.write_text(json.dumps({"coverage_end_utc": ""}), encoding="utf-8")
    result = _run(f'catalog_coverage_end "{report}" "2026-07-08T00:00:00Z"', tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == "2026-07-08T00:00:00Z"


def test_long_stale_listings_are_capped(tmp_path: Path) -> None:
    # A caller passing a glob can match dozens of orphaned artifacts from
    # datasets it never reads; a 40-line warning trains the operator to skip
    # warnings. The count stays exact, the listing does not.
    reference = _write(tmp_path / "fixture.csv", "a", 9_000.0)
    stale = [_write(tmp_path / f"old{i}.json", "b", 1_000.0 + i) for i in range(12)]
    quoted = " ".join(f'"{p}"' for p in stale)
    result = _run(f'require_fresh_inputs "{reference}" {quoted}', tmp_path)
    assert result.returncode == 0
    assert "staleness guard: 12 input(s) older" in result.stderr
    assert "... and 4 more" in result.stderr
    assert result.stderr.count("  stale ") == 8


def test_the_listing_cap_is_configurable(tmp_path: Path) -> None:
    reference = _write(tmp_path / "fixture.csv", "a", 9_000.0)
    stale = [_write(tmp_path / f"old{i}.json", "b", 1_000.0 + i) for i in range(5)]
    quoted = " ".join(f'"{p}"' for p in stale)
    result = _run(
        f'require_fresh_inputs "{reference}" {quoted}',
        tmp_path,
        env={"STALE_INPUTS_MAX_LISTED": "2"},
    )
    assert "... and 3 more" in result.stderr
    assert result.stderr.count("  stale ") == 2
