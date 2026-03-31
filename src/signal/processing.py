"""Top-level signal processing pipeline."""
from __future__ import annotations

from dataclasses import dataclass

from .calibrate import calibrate
from .filters import bandpass, notch
from .integrate import integrate_with_drift_correction
from .outliers import fix_dropouts
from .types import ProcessedSignal, RawSignal


@dataclass
class ProcessingConfig:
    """Configuration for signal processing pipeline."""

    # Calibration
    gain: float = 1.0
    offset: float = 0.0

    # Bandpass filter
    bandpass_low: float = 10.0  # Hz
    bandpass_high: float = 5000.0  # Hz

    # Notch filter
    notch_freqs: list[float] | None = None  # defaults to [60, 120, 180]
    notch_quality: float = 30.0

    # Sampling rate
    sample_rate: float = 100000.0  # Hz

    # Integration (Mirnov only)
    integrate_baseline_window: int = 50


def process_signal(raw: RawSignal, config: ProcessingConfig) -> ProcessedSignal:
    """Process a single raw signal through the full pipeline."""
    log: list[str] = []
    values = raw.values.copy()
    fs = config.sample_rate

    # 1. Fix dropouts
    values = fix_dropouts(values, raw.timestamps)
    log.append("dropout_fix")

    # 2. Calibrate
    values = calibrate(values, gain=config.gain, offset=config.offset)
    log.append("calibrate")

    # 3. Bandpass filter
    values = bandpass(values, fs, config.bandpass_low, config.bandpass_high)
    log.append("bandpass")

    # 4. Notch filter
    values = notch(values, fs, freqs=config.notch_freqs, quality=config.notch_quality)
    log.append("notch")

    # 5. Integration for Mirnov probes
    if raw.sensor_type == "mirnov":
        values = integrate_with_drift_correction(
            values, raw.timestamps, baseline_window=config.integrate_baseline_window
        )
        log.append("integrate")

    return ProcessedSignal(
        timestamps=raw.timestamps,
        values=values,
        channel_id=raw.channel_id,
        sensor_type=raw.sensor_type,
        processing_log=log,
    )


def process_shot(
    raw_signals: list[RawSignal], config: ProcessingConfig
) -> list[ProcessedSignal]:
    """Process all signals for a shot."""
    return [process_signal(sig, config) for sig in raw_signals]
