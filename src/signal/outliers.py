"""Outlier and dropout detection with interpolation."""
import numpy as np


def fix_dropouts(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Detect NaN values (dropouts) and linearly interpolate over them."""
    result = values.copy()
    nan_mask = np.isnan(result)
    if not nan_mask.any():
        return result
    good = ~nan_mask
    result[nan_mask] = np.interp(timestamps[nan_mask], timestamps[good], result[good])
    return result
