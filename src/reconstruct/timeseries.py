"""Time-series equilibrium reconstruction with warm-starting."""
from __future__ import annotations

from src.forward.sensors import SensorConfig

from .solver import ReconstructionResult, fit_equilibrium


def reconstruct_timeseries(
    mygs,
    measurement_series: list[dict[str, float]],
    sensor_config: SensorConfig,
    ip_targets: list[float] | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> list[ReconstructionResult]:
    """Reconstruct equilibrium for a series of time slices.

    Each slice is warm-started from the previous solution (the TokaMaker psi
    field is preserved between calls, giving the optimizer a head start).

    Args:
        mygs: Configured TokaMaker instance.
        measurement_series: List of measurement dicts, one per time slice.
        sensor_config: Sensor configuration.
        ip_targets: Optional list of Ip targets for each slice.
        max_iter: Maximum iterations per slice.
        tol: Convergence tolerance.

    Returns:
        List of ReconstructionResult, one per time slice.
    """
    results = []

    for i, measurements in enumerate(measurement_series):
        ip_target = ip_targets[i] if ip_targets is not None else None

        # For slices after the first, TokaMaker already has the previous
        # solution's psi field — this warm-starts the solve.
        # We skip init_psi (which would reset to a generic shape).
        result = fit_equilibrium(
            mygs,
            measurements,
            sensor_config,
            ip_target=ip_target,
            max_iter=max_iter,
            tol=tol,
            # After the first slice, we already have a good equilibrium
            # so we can skip the expensive Jacobian-based refinement
            # if the initial solve is close enough
            use_jacobian=(i == 0),
        )

        results.append(result)

    return results
