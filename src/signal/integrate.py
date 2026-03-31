"""Integration with drift correction for Mirnov probes (dB/dt -> B)."""
import numpy as np
from scipy.integrate import cumulative_trapezoid


def integrate_with_drift_correction(
    dbdt: np.ndarray, timestamps: np.ndarray, baseline_window: int = 50
) -> np.ndarray:
    """Integrate dB/dt to get B, with linear drift subtraction.

    Uses a two-step approach:
    1. Remove DC offset from dB/dt before integrating (prevents linear drift).
    2. Integrate the corrected dB/dt.
    3. Remove any residual linear trend from the integrated signal.

    Args:
        dbdt: dB/dt signal values.
        timestamps: Time array in seconds.
        baseline_window: Number of points at start/end for baseline estimation.
    """
    # Step 1: Remove DC offset from dB/dt to prevent dominant linear drift
    dbdt_corrected = dbdt - np.mean(dbdt)

    # Step 2: Integrate
    B = np.zeros_like(dbdt_corrected)
    B[1:] = cumulative_trapezoid(dbdt_corrected, timestamps)

    # Step 3: Remove residual linear trend using full-signal fit
    coeffs = np.polyfit(timestamps, B, 1)
    drift = np.polyval(coeffs, timestamps)
    return B - drift
