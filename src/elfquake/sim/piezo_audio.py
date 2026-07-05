"""Render simulated piezo sensor outputs as audio."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from elfquake.sim.piezo_signal import (
    dc_block as apply_dc_block,
    moving_average,
    read_piezo_rows,
    resample_for_audio,
    signal_by_step,
)


def render_piezo_audio(
    *,
    piezo_csv: Path,
    out_path: Path,
    sample_rate: int = 44100,
    duration_seconds: float = 20.0,
    gain: float = 0.95,
    smooth_steps: int = 64,
    sensor_id: int | None = None,
    dc_block: float = 0.0,
) -> dict[str, str]:
    if sample_rate < 8000:
        raise ValueError("sample_rate must be at least 8000")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if not 0 < gain <= 1:
        raise ValueError("gain must be greater than 0 and at most 1")
    if smooth_steps < 1:
        raise ValueError("smooth_steps must be at least 1")

    rows = read_piezo_rows(piezo_csv)
    steps, sensor_ids, signal = signal_by_step(rows, sensor_id=sensor_id)
    signal = apply_dc_block(signal, dc_block)
    sample_count = max(1, int(round(sample_rate * duration_seconds)))
    smoothed = moving_average(signal, smooth_steps)
    audio = resample_for_audio(smoothed, sample_count) * gain
    pcm = np.clip(audio * 32767, -32767, 32767).astype("<i2")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return {
        "piezo_file": str(piezo_csv),
        "audio_file": str(out_path),
        "audio_type": "sonified_sum_piezo_signal_by_step",
        "step_count": str(len(steps)),
        "sensor_count": str(len(sensor_ids)),
        "sample_rate_hz": str(sample_rate),
        "duration_seconds": f"{duration_seconds:.6f}",
        "audio_sample_count": str(sample_count),
        "smooth_steps": str(smooth_steps),
        "selected_sensor_id": "" if sensor_id is None else str(sensor_id),
        "dc_block": str(dc_block),
    }
