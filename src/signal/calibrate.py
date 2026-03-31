"""Calibration: gain correction and offset subtraction."""
import numpy as np


def calibrate(values: np.ndarray, gain: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Apply calibration: output = (values - offset) / gain."""
    return (values - offset) / gain
