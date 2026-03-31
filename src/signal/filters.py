"""Frequency-domain filters: bandpass and notch."""
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt


def bandpass(
    values: np.ndarray,
    fs: float,
    low_cutoff: float,
    high_cutoff: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter using SOS for numerical stability."""
    nyq = fs / 2.0
    low = low_cutoff / nyq
    high = high_cutoff / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, values)


def notch(
    values: np.ndarray,
    fs: float,
    freqs: list[float] | None = None,
    quality: float = 30.0,
) -> np.ndarray:
    """Apply notch filter at specified frequencies (default: 60 Hz + harmonics)."""
    if freqs is None:
        freqs = [60.0, 120.0, 180.0]
    result = values.copy()
    for freq in freqs:
        b, a = iirnotch(freq, quality, fs)
        # iirnotch is always 2nd order, so b/a form is stable
        from scipy.signal import filtfilt

        result = filtfilt(b, a, result)
    return result
