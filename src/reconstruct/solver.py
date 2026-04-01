"""Iterative equilibrium reconstruction solver."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.forward.sensors import SensorConfig
from src.store.schemas import EquilibriumResult

from .constraints import (
    estimate_ip_from_mirnov,
    forward_to_vector,
    measurements_to_vector,
    sensor_ids_ordered,
)


@dataclass
class ReconstructionDiagnostics:
    """Diagnostics from an equilibrium reconstruction."""

    chi_squared: float = 0.0
    per_sensor_residual: dict[str, float] = field(default_factory=dict)
    condition_number: float = 0.0
    n_iterations: int = 0
    converged: bool = False
    residual_history: list[float] = field(default_factory=list)


@dataclass
class ReconstructionResult:
    """Output of the reconstruction solver."""

    equilibrium: EquilibriumResult
    diagnostics: ReconstructionDiagnostics
    coil_currents: dict[str, float] = field(default_factory=dict)


def _get_boundary_rz(mygs, n_points: int = 100) -> tuple[list[float], list[float]]:
    """Extract plasma boundary (R, Z) coordinates from a solved equilibrium."""
    try:
        psi_boundary = mygs.psi_bounds[1]
        psi_axis = mygs.psi_bounds[0]
    except Exception:
        return [mygs.o_point[0]], [mygs.o_point[1]]

    # Trace boundary by evaluating psi on a fine grid and finding contour
    psi_eval = mygs.get_field_eval("psi")

    R_range = np.linspace(0.15, 0.50, 80)
    Z_range = np.linspace(-0.40, 0.40, 80)

    # Find boundary points by scanning radially at each Z
    boundary_r = []
    boundary_z = []

    for z in Z_range:
        psi_line = np.array([psi_eval.eval(np.array([r, z]))[0] for r in R_range])
        # Normalize
        if abs(psi_boundary - psi_axis) < 1e-12:
            continue
        psi_norm = (psi_line - psi_axis) / (psi_boundary - psi_axis)

        # Find crossings of psi_norm = 1.0 (boundary)
        for j in range(len(psi_norm) - 1):
            if (psi_norm[j] - 1.0) * (psi_norm[j + 1] - 1.0) < 0:
                # Linear interpolation for crossing
                frac = (1.0 - psi_norm[j]) / (psi_norm[j + 1] - psi_norm[j])
                r_cross = R_range[j] + frac * (R_range[j + 1] - R_range[j])
                boundary_r.append(float(r_cross))
                boundary_z.append(float(z))

    if not boundary_r:
        boundary_r = [mygs.o_point[0]]
        boundary_z = [mygs.o_point[1]]

    return boundary_r, boundary_z


def _build_sensitivity_matrix(
    mygs,
    sensor_config: SensorConfig,
    coil_names: list[str],
    base_currents: dict[str, float],
    base_synth: np.ndarray,
    delta_frac: float = 0.01,
) -> np.ndarray:
    """Build the Jacobian (sensitivity matrix) by finite differences on coil currents.

    Returns shape (n_sensors, n_coils).
    """
    n_sensors = sensor_config.n_total
    n_coils = len(coil_names)
    J = np.zeros((n_sensors, n_coils))

    for j, name in enumerate(coil_names):
        I_base = base_currents[name]
        delta = max(abs(I_base) * delta_frac, 1.0)  # at least 1 A-turn

        perturbed = dict(base_currents)
        perturbed[name] = I_base + delta
        mygs.set_coil_currents(perturbed)
        mygs.solve()
        synth_pert = forward_to_vector(mygs, sensor_config)

        J[:, j] = (synth_pert - base_synth) / delta

    # Restore original currents
    mygs.set_coil_currents(base_currents)
    mygs.solve()

    return J


def fit_equilibrium(
    mygs,
    measurements: dict[str, float],
    sensor_config: SensorConfig,
    ip_target: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
    use_jacobian: bool = True,
    damping: float = 0.5,
) -> ReconstructionResult:
    """Reconstruct the plasma equilibrium from sensor measurements.

    Uses an iterative approach:
    1. Estimate Ip from measurements (or use provided value)
    2. Solve equilibrium with TokaMaker's built-in constraint solver
    3. Compare forward model to measurements
    4. If not converged, adjust coil currents using sensitivity matrix
    5. Repeat until convergence

    Args:
        mygs: Configured TokaMaker instance (mesh, regions, profiles already set up).
               Should have coil bounds, regularization, and profiles configured.
        measurements: Dict of sensor_id -> measured value.
        sensor_config: Sensor configuration.
        ip_target: Plasma current target in Amps. Estimated from Mirnov data if None.
        max_iter: Maximum number of reconstruction iterations.
        tol: Convergence tolerance on chi-squared.
        use_jacobian: Whether to use Jacobian-based coil current refinement.
        damping: Damping factor for coil current updates (0-1).

    Returns:
        ReconstructionResult with equilibrium, diagnostics, and coil currents.
    """
    from OpenFUSIONToolkit.TokaMaker.util import create_isoflux

    meas_vec = measurements_to_vector(measurements, sensor_config)
    ids = sensor_ids_ordered(sensor_config)

    # Estimate Ip if not provided
    if ip_target is None:
        ip_target = estimate_ip_from_mirnov(measurements, sensor_config)

    # Set up TokaMaker constraints for CUTE geometry
    mygs.set_targets(Ip=ip_target, Ip_ratio=4.0)

    isoflux_pts = create_isoflux(80, 0.32, 0.0, 0.17, 1.7, 0.4)
    isoflux_pts = isoflux_pts[isoflux_pts[:, 0] > 0.3, :]
    isoflux_pts = np.vstack((isoflux_pts, np.array([[0.15, 0.0]])))
    x_points = np.array([[0.22, -0.33], [0.20, 0.34]])
    mygs.set_saddles(x_points)
    mygs.set_isoflux(np.vstack((isoflux_pts, x_points)))

    # Initialize psi and solve initial equilibrium
    mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
    mygs.solve()

    # Get coil names for sensitivity analysis
    coil_names = sorted(mygs.coil_sets.keys())

    diag = ReconstructionDiagnostics()
    J = None  # Jacobian (computed on demand)

    for iteration in range(max_iter):
        # Evaluate forward model
        synth_vec = forward_to_vector(mygs, sensor_config)

        # Compute residual
        residual = meas_vec - synth_vec
        chi_sq = float(np.sum(residual**2))
        diag.residual_history.append(chi_sq)

        if chi_sq < tol:
            diag.converged = True
            break

        # Check for stagnation
        if len(diag.residual_history) > 2:
            prev = diag.residual_history[-2]
            if abs(chi_sq - prev) / max(prev, 1e-12) < 1e-4:
                diag.converged = True
                break

        if not use_jacobian:
            # Without Jacobian, just do a single solve and accept the result
            diag.converged = True
            break

        # Build Jacobian (on first iteration or if convergence stalls)
        current_currents = mygs.get_coil_currents()[0]

        if J is None:
            J = _build_sensitivity_matrix(
                mygs, sensor_config, coil_names, current_currents, synth_vec
            )
            # Compute condition number
            try:
                diag.condition_number = float(np.linalg.cond(J))
            except Exception:
                diag.condition_number = 1.0

        # Solve damped least-squares for coil current update
        # (J^T J + lambda I)^{-1} J^T r
        JtJ = J.T @ J
        reg = 1e-6 * np.trace(JtJ) / len(coil_names) * np.eye(len(coil_names))
        delta_I = np.linalg.solve(JtJ + reg, J.T @ residual)

        # Apply damped update
        for j, name in enumerate(coil_names):
            current_currents[name] = current_currents[name] + damping * delta_I[j]

        mygs.set_coil_currents(current_currents)
        mygs.solve()

    diag.n_iterations = len(diag.residual_history)

    # Final forward model evaluation
    synth_vec = forward_to_vector(mygs, sensor_config)
    residual = meas_vec - synth_vec
    diag.chi_squared = float(np.sum(residual**2))
    diag.per_sensor_residual = {ids[i]: float(residual[i]) for i in range(len(ids))}

    if diag.condition_number == 0.0:
        diag.condition_number = 1.0

    # Extract equilibrium result
    stats = mygs.get_stats()
    boundary_r, boundary_z = _get_boundary_rz(mygs)

    equilibrium = EquilibriumResult(
        plasma_current=float(stats["Ip"]),
        q95=float(stats.get("q_95", 0.0)),
        beta_poloidal=float(stats.get("beta_pol", 0.0)),
        internal_inductance=float(stats.get("l_i", 0.0)),
        boundary_r=boundary_r,
        boundary_z=boundary_z,
    )

    coil_currents = mygs.get_coil_currents()[0]

    return ReconstructionResult(
        equilibrium=equilibrium,
        diagnostics=diag,
        coil_currents=dict(coil_currents),
    )
