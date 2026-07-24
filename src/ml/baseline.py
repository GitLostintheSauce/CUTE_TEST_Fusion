"""Classical iterative inversion baseline.

Recovers plasma parameters from a sensor vector by nonlinear least-squares
fitting of the same reduced forward model used to generate the data. This is
the fair, apples-to-apples baseline the learned surrogate is benchmarked
against: both invert the identical physics, so any speed difference is due to
the method (amortized network evaluation vs. per-shot iterative optimization),
not a different problem.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from src.ml.dataset import PARAM_NAMES, PARAM_RANGES, SensorLayout, forward_signals


def invert_least_squares(
    signals: np.ndarray,
    layout: SensorLayout,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Recover [Ip, R0, Z0, a] from a sensor vector by nonlinear least squares.

    Args:
        signals: Observed 130-element sensor vector.
        layout: Sensor layout used by the forward model.
        x0: Optional initial guess; defaults to the range midpoints.

    Returns:
        Estimated parameter vector [Ip, R0, Z0, a].
    """
    lo = np.array([PARAM_RANGES[n][0] for n in PARAM_NAMES])
    hi = np.array([PARAM_RANGES[n][1] for n in PARAM_NAMES])
    if x0 is None:
        x0 = 0.5 * (lo + hi)

    # Scale parameters to O(1) so the optimizer is well-conditioned.
    span = hi - lo

    def residual(x_scaled):
        x = lo + x_scaled * span
        pred = forward_signals(x[0], x[1], x[2], x[3], layout)
        return pred - signals

    res = least_squares(
        residual, (x0 - lo) / span, bounds=(0.0, 1.0), method="trf",
        xtol=1e-10, ftol=1e-10,
    )
    return lo + res.x * span
