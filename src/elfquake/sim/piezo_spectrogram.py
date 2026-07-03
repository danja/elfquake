"""Render VLF-like spectrograms from piezo sensor simulation outputs."""

from __future__ import annotations

import csv
import json
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from elfquake.sim.heatmap import _grid_to_rgb


def render_piezo_spectrogram(
    *,
    piezo_csv: Path,
    out_path: Path,
    metadata_out: Path | None = None,
    start_time_utc: str = "2026-01-01T00:00:00Z",
    step_seconds: int = 60,
    freq_min: float = 0.0,
    freq_max: float | None = None,
    freq_bins: int = 96,
    window_steps: int = 64,
    scale: int = 4,
    gamma: float = 0.85,
    sensor_id: int | None = None,
    dc_block: float = 0.0,
) -> dict[str, str]:
    if step_seconds < 1:
        raise ValueError("step_seconds must be at least 1")
    if freq_min < 0:
        raise ValueError("freq_min must be non-negative")
    if freq_bins < 2:
        raise ValueError("freq_bins must be at least 2")
    if window_steps < 2:
        raise ValueError("window_steps must be at least 2")
    if scale < 1:
        raise ValueError("scale must be at least 1")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - depends on optional environment.
        raise RuntimeError("Pillow is required for piezo spectrogram rendering") from error

    rows = _read_rows(piezo_csv)
    steps, sensor_ids, signal = _signal_by_step(rows, sensor_id=sensor_id)
    signal = _dc_block(signal, dc_block)
    sample_rate_hz = 1.0 / step_seconds
    nyquist_hz = sample_rate_hz / 2.0
    resolved_freq_max = nyquist_hz if freq_max is None else min(freq_max, nyquist_hz)
    if not freq_min < resolved_freq_max:
        raise ValueError("freq_min must be below the Nyquist-limited freq_max")

    power = _stft_power(
        signal=signal,
        step_seconds=step_seconds,
        freq_min=freq_min,
        freq_max=resolved_freq_max,
        freq_bins=freq_bins,
        window_steps=window_steps,
    )

    display = np.log1p(power)
    image_grid = display[::-1, :]
    rgb = _grid_to_rgb(image_grid, color_min=0.0, gamma=gamma)
    image = Image.fromarray(rgb, mode="RGB")
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    start = _parse_utc(start_time_utc)
    end = start + timedelta(seconds=(signal.size - 1 if signal.size else 0) * step_seconds)
    report = {
        "piezo_file": str(piezo_csv),
        "spectrogram_file": str(out_path),
        "time_start_utc": _format_utc(start),
        "time_end_utc": _format_utc(end),
        "step_count": str(len(steps)),
        "sample_count": str(signal.size),
        "step_seconds": str(step_seconds),
        "sample_rate_hz": f"{sample_rate_hz:.12f}",
        "nyquist_hz": f"{nyquist_hz:.12f}",
        "sensor_count": str(len(sensor_ids)),
        "freq_min_hz": f"{freq_min:.12f}",
        "freq_max_hz": f"{resolved_freq_max:.12f}",
        "freq_bins": str(freq_bins),
        "window_steps": str(window_steps),
        "frequency_axis": "fft_from_step_seconds",
        "source_signal": "sum_piezo_signal_by_step",
        "selected_sensor_id": "" if sensor_id is None else str(sensor_id),
        "dc_block": str(dc_block),
        "width_px": str(image.width),
        "height_px": str(image.height),
        "max_power": f"{float(power.max()) if power.size else 0.0:.9f}",
    }
    if metadata_out is not None:
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def render_piezo_timeseries_spectrogram(
    *,
    piezo_csv: Path,
    out_path: Path,
    metadata_out: Path | None = None,
    start_time_utc: str = "2026-01-01T00:00:00Z",
    step_seconds: int = 60,
    freq_min: float = 0.0,
    freq_max: float | None = None,
    freq_bins: int = 96,
    window_steps: int = 64,
    scale: int = 4,
    gamma: float = 0.85,
    timeseries_height: int = 48,
    output_width: int = 1600,
    sensor_id: int | None = None,
    dc_block: float = 0.0,
) -> dict[str, str]:
    if timeseries_height < 16:
        raise ValueError("timeseries_height must be at least 16")
    if output_width < 1:
        raise ValueError("output_width must be at least 1")
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - depends on optional environment.
        raise RuntimeError("Pillow is required for piezo rendering") from error

    rows = _read_rows(piezo_csv)
    steps, sensor_ids, signal = _signal_by_step(rows, sensor_id=sensor_id)
    signal = _dc_block(signal, dc_block)
    sample_rate_hz = 1.0 / step_seconds
    nyquist_hz = sample_rate_hz / 2.0
    resolved_freq_max = nyquist_hz if freq_max is None else min(freq_max, nyquist_hz)
    if not freq_min < resolved_freq_max:
        raise ValueError("freq_min must be below the Nyquist-limited freq_max")

    power = _stft_power(
        signal=signal,
        step_seconds=step_seconds,
        freq_min=freq_min,
        freq_max=resolved_freq_max,
        freq_bins=freq_bins,
        window_steps=window_steps,
    )
    display_signal, display_power = _compress_for_display(signal=signal, power=power, output_width=output_width)
    width = max(1, display_signal.size)
    top = Image.new("RGB", (width, timeseries_height), (6, 10, 24))
    draw = ImageDraw.Draw(top)
    if display_signal.size:
        normalized = _normalize_audio_signal(display_signal)
        mid = timeseries_height // 2
        amplitude = max(1, timeseries_height // 2 - 4)
        points = [
            (index, int(round(mid - value * amplitude)))
            for index, value in enumerate(normalized)
        ]
        if len(points) == 1:
            x, y = points[0]
            draw.line([(x, mid), (x, y)], fill=(246, 238, 210))
        else:
            draw.line(points, fill=(246, 238, 210), width=1)
        draw.line([(0, mid), (width - 1, mid)], fill=(32, 92, 122))

    display = np.log1p(display_power)
    spectrogram_rgb = _grid_to_rgb(display[::-1, :], color_min=0.0, gamma=gamma)
    spectrogram = Image.fromarray(spectrogram_rgb, mode="RGB")
    spacer = Image.new("RGB", (width, 2), (0, 0, 0))
    combined = Image.new("RGB", (width, timeseries_height + 2 + spectrogram.height), (0, 0, 0))
    combined.paste(top, (0, 0))
    combined.paste(spacer, (0, timeseries_height))
    combined.paste(spectrogram, (0, timeseries_height + 2))
    if scale > 1:
        combined = combined.resize((combined.width * scale, combined.height * scale), Image.Resampling.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)

    start = _parse_utc(start_time_utc)
    end = start + timedelta(seconds=(signal.size - 1 if signal.size else 0) * step_seconds)
    report = {
        "piezo_file": str(piezo_csv),
        "image_file": str(out_path),
        "plot_type": "timeseries_plus_fft_spectrogram",
        "time_start_utc": _format_utc(start),
        "time_end_utc": _format_utc(end),
        "step_count": str(len(steps)),
        "sample_count": str(signal.size),
        "step_seconds": str(step_seconds),
        "sample_rate_hz": f"{sample_rate_hz:.12f}",
        "nyquist_hz": f"{nyquist_hz:.12f}",
        "sensor_count": str(len(sensor_ids)),
        "freq_min_hz": f"{freq_min:.12f}",
        "freq_max_hz": f"{resolved_freq_max:.12f}",
        "freq_bins": str(freq_bins),
        "window_steps": str(window_steps),
        "frequency_axis": "fft_from_step_seconds",
        "source_signal": "sum_piezo_signal_by_step",
        "selected_sensor_id": "" if sensor_id is None else str(sensor_id),
        "dc_block": str(dc_block),
        "width_px": str(combined.width),
        "height_px": str(combined.height),
        "timeseries_height_px": str(timeseries_height * scale),
        "display_sample_count": str(display_signal.size),
        "display_decimation": str(max(1, int(np.ceil(signal.size / max(display_signal.size, 1))))),
        "max_power": f"{float(power.max()) if power.size else 0.0:.9f}",
    }
    if metadata_out is not None:
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


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

    rows = _read_rows(piezo_csv)
    steps, sensor_ids, signal = _signal_by_step(rows, sensor_id=sensor_id)
    signal = _dc_block(signal, dc_block)
    sample_count = max(1, int(round(sample_rate * duration_seconds)))
    smoothed = _moving_average(signal, smooth_steps)
    audio = _resample_for_audio(smoothed, sample_count) * gain
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


def render_piezo_strain_vlf_summary(
    *,
    piezo_csv: Path,
    out_path: Path,
    metadata_out: Path | None = None,
    start_time_utc: str = "2026-01-01T00:00:00Z",
    carrier_freq_min_hz: float = 0.0,
    carrier_freq_max_hz: float = 24000.0,
    freq_bins: int = 192,
    scale: int = 4,
    gamma: float = 0.85,
    timeseries_height: int = 48,
    output_width: int = 1600,
    sensor_id: int | None = None,
    dc_block: float = 0.995,
) -> dict[str, str]:
    """Render a VLF-shaped analogue display from the simulated piezo strain envelope.

    This is not an FFT of the simulation time series. The simulation supplies a slow
    strain/release envelope, and this renderer maps it onto deterministic analogue
    carrier bands so the image has the same kind of shape expected from VLF
    receiver data.
    """
    if carrier_freq_min_hz < 0:
        raise ValueError("carrier_freq_min_hz must be non-negative")
    if not carrier_freq_min_hz < carrier_freq_max_hz:
        raise ValueError("carrier_freq_min_hz must be below carrier_freq_max_hz")
    if freq_bins < 8:
        raise ValueError("freq_bins must be at least 8")
    if scale < 1:
        raise ValueError("scale must be at least 1")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if timeseries_height < 16:
        raise ValueError("timeseries_height must be at least 16")
    if output_width < 1:
        raise ValueError("output_width must be at least 1")
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - depends on optional environment.
        raise RuntimeError("Pillow is required for piezo rendering") from error

    rows = _read_rows(piezo_csv)
    steps, sensor_ids, signal = _signal_by_step(rows, sensor_id=sensor_id)
    signal = _dc_block(signal, dc_block)
    power = _strain_vlf_power(
        signal=signal,
        freq_bins=freq_bins,
        carrier_freq_min_hz=carrier_freq_min_hz,
        carrier_freq_max_hz=carrier_freq_max_hz,
    )
    display_signal, display_power = _compress_for_display(
        signal=signal,
        power=power,
        output_width=output_width,
    )

    width = max(1, display_signal.size)
    top = Image.new("RGB", (width, timeseries_height), (6, 10, 24))
    draw = ImageDraw.Draw(top)
    if display_signal.size:
        normalized = _normalize_audio_signal(display_signal)
        mid = timeseries_height // 2
        amplitude = max(1, timeseries_height // 2 - 4)
        points = [
            (index, int(round(mid - value * amplitude)))
            for index, value in enumerate(normalized)
        ]
        if len(points) == 1:
            x, y = points[0]
            draw.line([(x, mid), (x, y)], fill=(246, 238, 210))
        else:
            draw.line(points, fill=(246, 238, 210), width=1)
        draw.line([(0, mid), (width - 1, mid)], fill=(32, 92, 122))

    display = np.log1p(display_power)
    spectrogram_rgb = _grid_to_rgb(display[::-1, :], color_min=0.0, gamma=gamma)
    spectrogram = Image.fromarray(spectrogram_rgb, mode="RGB")
    spacer = Image.new("RGB", (width, 2), (0, 0, 0))
    combined = Image.new("RGB", (width, timeseries_height + 2 + spectrogram.height), (0, 0, 0))
    combined.paste(top, (0, 0))
    combined.paste(spacer, (0, timeseries_height))
    combined.paste(spectrogram, (0, timeseries_height + 2))
    if scale > 1:
        combined = combined.resize((combined.width * scale, combined.height * scale), Image.Resampling.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)

    start = _parse_utc(start_time_utc)
    report = {
        "piezo_file": str(piezo_csv),
        "image_file": str(out_path),
        "plot_type": "strain_envelope_vlf_analogue",
        "time_start_utc": _format_utc(start),
        "step_count": str(len(steps)),
        "sample_count": str(signal.size),
        "sensor_count": str(len(sensor_ids)),
        "selected_sensor_id": "" if sensor_id is None else str(sensor_id),
        "dc_block": str(dc_block),
        "carrier_freq_min_hz": f"{carrier_freq_min_hz:.6f}",
        "carrier_freq_max_hz": f"{carrier_freq_max_hz:.6f}",
        "freq_bins": str(freq_bins),
        "frequency_axis": "analogue_vlf_carrier_hz",
        "source_signal": "piezo_strain_release_envelope",
        "width_px": str(combined.width),
        "height_px": str(combined.height),
        "timeseries_height_px": str(timeseries_height * scale),
        "display_sample_count": str(display_signal.size),
        "display_decimation": str(max(1, int(np.ceil(signal.size / max(display_signal.size, 1))))),
        "max_power": f"{float(power.max()) if power.size else 0.0:.9f}",
    }
    if metadata_out is not None:
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _stft_power(
    *,
    signal: np.ndarray,
    step_seconds: int,
    freq_min: float,
    freq_max: float,
    freq_bins: int,
    window_steps: int,
) -> np.ndarray:
    if signal.size == 0:
        return np.zeros((freq_bins, 0), dtype=np.float64)

    fft_freqs = np.fft.rfftfreq(window_steps, d=step_seconds)
    target_freqs = np.linspace(freq_min, freq_max, freq_bins)
    power = np.zeros((freq_bins, signal.size), dtype=np.float64)
    window = _analysis_window(window_steps)

    for index in range(signal.size):
        start = max(0, index - window_steps + 1)
        segment = signal[start : index + 1]
        padded = np.zeros(window_steps, dtype=np.float64)
        padded[-segment.size :] = segment
        transformed = np.fft.rfft(padded * window)
        raw_power = np.abs(transformed) ** 2
        power[:, index] = np.interp(target_freqs, fft_freqs, raw_power)
    return power


def _signal_by_step(rows: list[dict[str, str]], *, sensor_id: int | None = None) -> tuple[list[int], list[int], np.ndarray]:
    steps = sorted({int(row["step"]) for row in rows})
    sensor_ids = sorted({int(row["sensor_id"]) for row in rows})
    sample_count = (max(steps) - min(steps) + 1) if steps else 0
    signal = np.zeros(sample_count, dtype=np.float64)
    if steps:
        first_step = min(steps)
        for row in rows:
            if sensor_id is not None and int(row["sensor_id"]) != sensor_id:
                continue
            signal[int(row["step"]) - first_step] += float(row["piezo_signal"])
    return steps, sensor_ids, signal


def _dc_block(signal: np.ndarray, coefficient: float) -> np.ndarray:
    if signal.size == 0 or coefficient <= 0:
        return signal
    if not 0 < coefficient < 1:
        raise ValueError("dc_block must be between 0 and 1")
    output = np.zeros_like(signal, dtype=np.float64)
    previous_input = 0.0
    previous_output = 0.0
    for index, value in enumerate(signal.astype(np.float64)):
        current = value - previous_input + coefficient * previous_output
        output[index] = current
        previous_input = value
        previous_output = current
    return output


def _resample_for_audio(signal: np.ndarray, sample_count: int) -> np.ndarray:
    if signal.size == 0:
        return np.zeros(sample_count, dtype=np.float64)
    if signal.size == 1:
        source = np.repeat(signal, 2)
    else:
        source = signal
    normalized = _normalize_audio_signal(source)
    x_old = np.linspace(0.0, 1.0, normalized.size)
    x_new = np.linspace(0.0, 1.0, sample_count)
    return np.interp(x_new, x_old, normalized)


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if signal.size == 0 or window <= 1:
        return signal.astype(np.float64)
    resolved = min(window, signal.size)
    kernel = np.ones(resolved, dtype=np.float64) / resolved
    return np.convolve(signal.astype(np.float64), kernel, mode="same")


def _compress_for_display(
    *,
    signal: np.ndarray,
    power: np.ndarray,
    output_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if signal.size <= output_width:
        return signal, power
    edges = np.linspace(0, signal.size, output_width + 1, dtype=np.int64)
    display_signal = np.zeros(output_width, dtype=np.float64)
    display_power = np.zeros((power.shape[0], output_width), dtype=np.float64)
    for index in range(output_width):
        start = int(edges[index])
        end = max(start + 1, int(edges[index + 1]))
        segment = signal[start:end]
        # Preserve spikes while keeping polarity after centering.
        mean = float(segment.mean()) if segment.size else 0.0
        peak = float(segment[np.argmax(np.abs(segment - mean))]) if segment.size else mean
        display_signal[index] = peak
        if power.shape[1]:
            display_power[:, index] = power[:, start:end].max(axis=1)
    return display_signal, display_power


def _normalize_audio_signal(signal: np.ndarray) -> np.ndarray:
    if signal.size == 0:
        return signal.astype(np.float64)
    centered = signal.astype(np.float64) - float(signal.mean())
    peak = float(np.max(np.abs(centered))) if centered.size else 0.0
    if peak <= 0:
        return np.zeros_like(centered, dtype=np.float64)
    return centered / peak


def _analysis_window(size: int) -> np.ndarray:
    if size <= 2:
        return np.ones(size, dtype=np.float64)
    return np.hanning(size + 2)[1:-1]


def _strain_vlf_power(
    *,
    signal: np.ndarray,
    freq_bins: int,
    carrier_freq_min_hz: float,
    carrier_freq_max_hz: float,
) -> np.ndarray:
    if signal.size == 0:
        return np.zeros((freq_bins, 0), dtype=np.float64)
    envelope = np.abs(signal.astype(np.float64))
    if envelope.size > 3:
        envelope = _moving_average(envelope, min(5, envelope.size))
    robust_peak = float(np.quantile(envelope, 0.99)) if envelope.size else 0.0
    if robust_peak <= 0:
        normalized = np.zeros_like(envelope, dtype=np.float64)
    else:
        normalized = np.clip(envelope / robust_peak, 0.0, 2.0)

    freqs = np.linspace(carrier_freq_min_hz, carrier_freq_max_hz, freq_bins)
    span = max(carrier_freq_max_hz - carrier_freq_min_hz, 1.0)
    centers = carrier_freq_min_hz + span * np.array([0.05, 0.09, 0.14, 0.22, 0.34, 0.48, 0.65, 0.82])
    widths = span * np.array([0.010, 0.008, 0.012, 0.015, 0.018, 0.020, 0.018, 0.014])
    power = np.zeros((freq_bins, signal.size), dtype=np.float64)
    previous = np.zeros(freq_bins, dtype=np.float64)
    freq_tilt = 1.0 / (1.0 + 4.0 * ((freqs - carrier_freq_min_hz) / span))

    previous_amplitude = 0.0
    for index, amplitude in enumerate(normalized):
        onset = max(0.0, amplitude - previous_amplitude)
        column = 0.020 * freq_tilt
        burst = amplitude ** 1.15
        column += 0.24 * burst * freq_tilt
        for band_index, center in enumerate(centers):
            shimmer = 0.72 + 0.28 * np.sin(index * (0.019 + band_index * 0.004) + band_index * 1.7)
            band = np.exp(-0.5 * ((freqs - center) / max(widths[band_index], 1.0)) ** 2)
            column += burst * shimmer * (0.75 + 0.08 * band_index) * band
        if amplitude > 0.45:
            wide_center = carrier_freq_min_hz + span * (0.35 + 0.25 * np.sin(index * 0.037))
            wide = np.exp(-0.5 * ((freqs - wide_center) / (span * 0.18)) ** 2)
            column += (amplitude - 0.45) * 1.10 * wide
        if amplitude > 0.70 or onset > 0.10:
            column += (0.45 * amplitude + 1.25 * onset) * (0.55 + 0.45 * freq_tilt)
        previous = np.maximum(column, previous * 0.80)
        power[:, index] = previous
        previous_amplitude = amplitude
    return power


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
