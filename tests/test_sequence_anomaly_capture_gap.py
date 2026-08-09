"""Capture-gap guard for the label-free VLF anomaly scorer.

Regression cover for the 2026-08-04 Pisa retrospective, where the two
highest-scoring days in the whole Cumiana record turned out to be windows whose
24-frame lookback straddled a 13-day collector outage.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from elfquake.models.torch_self_supervised import (
    _descriptor_window_gap_seconds,
    _median_step_seconds,
)
from elfquake.models.torch_sequence_data import SequenceDataset


def _dataset(times: list[datetime]) -> SequenceDataset:
    return SequenceDataset(
        dataset_id="cumiana",
        modality="real_vlf_image",
        time_to_index={t.strftime("%Y-%m-%dT%H:%M:%SZ"): i for i, t in enumerate(times)},
        values=[[float(i)] for i in range(len(times))],
        feature_names=["intensity"],
    )


def _regular(count: int, *, start: datetime, minutes: int = 30) -> list[datetime]:
    return [start + timedelta(minutes=minutes * i) for i in range(count)]


def test_median_step_ignores_the_outage() -> None:
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    times = _regular(20, start=start)
    times += _regular(20, start=times[-1] + timedelta(days=13))
    assert _median_step_seconds(_dataset(times)) == 1800.0


def test_windows_spanning_a_gap_are_isolated_and_scores_recover() -> None:
    lookback = 24
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    before = _regular(30, start=start)
    after = _regular(30, start=before[-1] + timedelta(days=13))
    dataset = _dataset(before + after)

    gaps = _descriptor_window_gap_seconds(dataset, lookback_steps=lookback, stride=1)
    threshold = lookback * _median_step_seconds(dataset)
    flagged = [index for index, gap in enumerate(gaps) if gap > threshold]

    # Exactly the windows still holding a pre-gap frame are flagged, and the
    # flag clears as a step change once the lookback buffer flushes.
    assert flagged, "the outage must be detected"
    assert flagged == list(range(flagged[0], flagged[-1] + 1)), "flagged block must be contiguous"
    assert len(flagged) == lookback - 1
    assert gaps[flagged[-1] + 1] == 1800.0


def test_continuous_capture_flags_nothing() -> None:
    dataset = _dataset(_regular(60, start=datetime(2026, 7, 1, tzinfo=timezone.utc)))
    gaps = _descriptor_window_gap_seconds(dataset, lookback_steps=24, stride=1)
    threshold = 24 * _median_step_seconds(dataset)
    assert threshold > 0
    assert not [gap for gap in gaps if gap > threshold]


def test_gap_shorter_than_the_window_is_tolerated() -> None:
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    times = _regular(30, start=start)
    # A single missed capture is normal jitter, not an outage.
    times += _regular(30, start=times[-1] + timedelta(minutes=90))
    dataset = _dataset(times)
    gaps = _descriptor_window_gap_seconds(dataset, lookback_steps=24, stride=1)
    threshold = 24 * _median_step_seconds(dataset)
    assert not [gap for gap in gaps if gap > threshold]
