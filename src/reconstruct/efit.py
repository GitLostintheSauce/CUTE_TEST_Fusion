"""EFIT-style equilibrium reconstruction using Green's function matrix.

Reconstructs the plasma equilibrium from sensor measurements by decomposing
the measured fields into vacuum (coil) and plasma contributions:

    y_measured = G_coils @ I_coils + y_plasma

The algorithm iterates:
1. Estimate coil currents via regularized least-squares: I = (G'G + λI)^{-1} G' (y - y_plasma)
2. Set coil currents in TokaMaker and solve GS equation
3. Compute new plasma contribution
4. Repeat until convergence
"""
from __future__ import annotations

import numpy as np

from src.forward.sensors import SensorConfig
from src.reconstruct.constraints import (
    estimate_ip_from_mirnov,
    forward_to_vector,
    measurements_to_vector,
    sensor_ids_ordered,
)
from src.reconstruct.solver import (
    ReconstructionDiagnostics,
    ReconstructionResult,
    _get_boundary_rz,
)
from src.store.schemas import EquilibriumResult


def _tikhonov_solve(
    G: np.ndarray,
    y: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Solve the Tikhonov-regularized least-squares problem.

    Minimizes ||G x - y||^2 + λ ||x||^2.

    Returns:
        x: Solution vector.
    """
    GtG = G.T @ G
    Gty = G.T @ y
    n = GtG.shape[0]
    return np.linalg.solve(GtG + lam * np.eye(n), Gty)


def select_regularization(G: np.ndarray, y: np.ndarray) -> float:
    """Select Tikhonov regularization parameter based on SVD.

    Sets λ to a small fraction of the minimum singular value squared,
    providing stability without over-regularizing.
    """
    s = np.linalg.svd(G, compute_uv=False)
    # cond(G'G + λI) = (σ_max^2 + λ) / (σ_min^2 + λ) < target_cond
    # → λ > (σ_max^2 - target_cond * σ_min^2) / (target_cond - 1)
    target_cond = 5e5
    lam_floor = max(0.0, (s[0] ** 2 - target_cond * s[-1] ** 2) / (target_cond - 1))
    return max(lam_floor, 1e-16)


def efit_reconstruct(
    mygs,
    measurements: dict[str, float],
    sensor_config: SensorConfig,
    G: np.ndarray,
    coil_names: list[str],
    max_iter: int = 30,
    tol: float = 1e-8,
    lam: float | None = None,
) -> ReconstructionResult:
    """EFIT-style reconstruction from measurements using Green's functions.

    Args:
        mygs: Configured TokaMaker instance (mesh, regions, profiles set).
              Must have a previously solved equilibrium (call _solve_reference first).
        measurements: Dict of sensor_id -> measured value.
        sensor_config: Sensor configuration.
        G: Green's function matrix, shape (n_sensors, n_coils).
        coil_names: Ordered coil names matching G columns.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on chi-squared.
        lam: Tikhonov regularization parameter. Auto-selected if None.

    Returns:
        ReconstructionResult with equilibrium, diagnostics, and coil currents.
    """
    from OpenFUSIONToolkit.TokaMaker.util import create_isoflux, create_power_flux_fun

    y_meas = measurements_to_vector(measurements, sensor_config)
    ids = sensor_ids_ordered(sensor_config)
    n_coils = len(coil_names)

    if lam is None:
        lam = select_regularization(G, y_meas)

    # Condition number of regularized system
    GtG = G.T @ G
    reg_matrix = GtG + lam * np.eye(n_coils)
    cond_number = float(np.linalg.cond(reg_matrix))

    # Use Ip from current (reference) equilibrium as starting point
    try:
        ip_estimate = float(mygs.get_stats()["Ip"])
    except Exception:
        ip_estimate = estimate_ip_from_mirnov(measurements, sensor_config)

    # Set up profiles and shape constraints for GS solve
    ffp_prof = create_power_flux_fun(40, 1.5, 2.0)
    pp_prof = create_power_flux_fun(40, 4.0, 1.0)
    mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)

    isoflux_pts = create_isoflux(80, 0.32, 0.0, 0.17, 1.7, 0.4)
    isoflux_pts = isoflux_pts[isoflux_pts[:, 0] > 0.3, :]
    isoflux_pts = np.vstack((isoflux_pts, np.array([[0.15, 0.0]])))
    x_points = np.array([[0.22, -0.33], [0.20, 0.34]])
    mygs.set_saddles(x_points)
    mygs.set_isoflux(np.vstack((isoflux_pts, x_points)))

    diag = ReconstructionDiagnostics()
    diag.condition_number = cond_number

    # Get initial plasma contribution from the current (reference) solve
    y_total_init = forward_to_vector(mygs, sensor_config)
    # Get current coil currents from the reference solve
    coil_currents_ref = mygs.get_coil_currents()[0]
    I_ref = np.array([coil_currents_ref[name] for name in coil_names])
    y_plasma = y_total_init - G @ I_ref

    # Initial coil current estimate: fit from measurements minus plasma estimate
    I_coils = _tikhonov_solve(G, y_meas - y_plasma, lam)

    for iteration in range(max_iter):
        current_dict = {coil_names[j]: float(I_coils[j]) for j in range(n_coils)}
        mygs.set_coil_currents(current_dict)

        # Use Ip from previous solve after first iteration
        if iteration > 0:
            try:
                stats = mygs.get_stats()
                ip_estimate = float(stats["Ip"])
            except Exception:
                pass

        mygs.set_targets(Ip=ip_estimate, Ip_ratio=4.0)
        mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)

        try:
            mygs.solve()
        except ValueError:
            # If solve fails, try with slightly different Ip
            mygs.set_targets(Ip=ip_estimate * 0.9, Ip_ratio=4.0)
            mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
            try:
                mygs.solve()
            except ValueError:
                break

        # Compute forward model and update plasma contribution
        y_total = forward_to_vector(mygs, sensor_config)
        y_plasma = y_total - G @ I_coils

        # Re-fit coil currents
        I_coils = _tikhonov_solve(G, y_meas - y_plasma, lam)

        # Check convergence
        residual = y_meas - y_total
        chi_sq = float(np.sum(residual**2))
        diag.residual_history.append(chi_sq)

        if chi_sq < tol:
            diag.converged = True
            break

        if len(diag.residual_history) > 2:
            prev = diag.residual_history[-2]
            if abs(chi_sq - prev) / max(prev, 1e-12) < 1e-4:
                diag.converged = True
                break

    diag.n_iterations = len(diag.residual_history)

    # Final evaluation
    y_synth = forward_to_vector(mygs, sensor_config)
    residual = y_meas - y_synth
    diag.chi_squared = float(np.sum(residual**2))
    diag.per_sensor_residual = {ids[i]: float(residual[i]) for i in range(len(ids))}

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

    final_currents = {coil_names[j]: float(I_coils[j]) for j in range(n_coils)}

    return ReconstructionResult(
        equilibrium=equilibrium,
        diagnostics=diag,
        coil_currents=final_currents,
    )
