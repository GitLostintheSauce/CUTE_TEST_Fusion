"""Sensor placement optimization for CUTE reconstruction.

Determines which sensor locations maximize reconstruction accuracy using
the Green's function matrix. Provides:
- Fisher information analysis
- Leave-one-out importance ranking
- Greedy forward selection of minimum viable sensor set
- Sensor type complementarity analysis
"""
from __future__ import annotations

import numpy as np

from src.forward.sensors import SensorConfig


def fisher_information(
    G: np.ndarray,
    noise_std: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Compute the Fisher information matrix F = G^T W G.

    Args:
        G: Green's function matrix, shape (n_sensors, n_coils).
        noise_std: Sensor noise standard deviation (scalar or per-sensor array).

    Returns:
        F: Fisher information matrix, shape (n_coils, n_coils).
    """
    if np.isscalar(noise_std):
        W = np.eye(G.shape[0]) / noise_std**2  # type: ignore[operator]
    else:
        W = np.diag(1.0 / np.array(noise_std)**2)
    return G.T @ W @ G


def parameter_uncertainty(G: np.ndarray, noise_std: float = 1.0) -> np.ndarray:
    """Compute parameter uncertainty from Cramér-Rao bound.

    Returns the diagonal of (G^T W G)^{-1}, which gives the minimum
    variance for each parameter (coil current) estimate.

    Args:
        G: Green's function matrix, shape (n_sensors, n_coils).
        noise_std: Sensor noise standard deviation.

    Returns:
        uncertainties: shape (n_coils,), diagonal of inverse Fisher matrix.
    """
    F = fisher_information(G, noise_std)
    try:
        F_inv = np.linalg.inv(F)
        return np.diag(F_inv)
    except np.linalg.LinAlgError:
        return np.full(G.shape[1], np.inf)


def reconstruction_error_metric(G: np.ndarray, noise_std: float = 1.0) -> float:
    """Compute a scalar reconstruction error metric from G.

    Uses the trace of (G^T G)^{-1} as a proxy for total reconstruction error
    (A-optimality criterion).

    Args:
        G: Green's function matrix (subset of sensors).
        noise_std: Sensor noise standard deviation.

    Returns:
        error: Scalar reconstruction error metric.
    """
    F = fisher_information(G, noise_std)
    try:
        F_inv = np.linalg.inv(F)
        return float(np.trace(F_inv))
    except np.linalg.LinAlgError:
        return float("inf")


def leave_one_out(
    G: np.ndarray,
    sensor_ids: list[str],
    noise_std: float = 1.0,
) -> dict[str, float]:
    """Leave-one-out analysis: how much does removing each sensor hurt?

    For each sensor, removes it from G, computes reconstruction error,
    and records the degradation compared to the full set.

    Args:
        G: Full Green's function matrix, shape (n_sensors, n_coils).
        sensor_ids: Ordered sensor identifiers.
        noise_std: Sensor noise standard deviation.

    Returns:
        degradation: Dict of sensor_id -> error increase when removed.
    """
    baseline_error = reconstruction_error_metric(G, noise_std)
    degradation = {}

    for i, sid in enumerate(sensor_ids):
        G_reduced = np.delete(G, i, axis=0)
        error_reduced = reconstruction_error_metric(G_reduced, noise_std)
        degradation[sid] = error_reduced - baseline_error

    return degradation


def greedy_forward_selection(
    G: np.ndarray,
    sensor_ids: list[str],
    max_sensors: int | None = None,
    noise_std: float = 1.0,
) -> tuple[list[str], list[float]]:
    """Greedy forward selection of sensors.

    Starts with no sensors and iteratively adds the one that most reduces
    reconstruction error.

    Args:
        G: Full Green's function matrix, shape (n_sensors, n_coils).
        sensor_ids: Ordered sensor identifiers.
        max_sensors: Maximum number of sensors to select. Default: all.
        noise_std: Sensor noise standard deviation.

    Returns:
        selected_ids: Ordered list of selected sensor IDs.
        errors: Reconstruction error after each addition.
    """
    n_sensors, n_coils = G.shape
    if max_sensors is None:
        max_sensors = n_sensors

    available = set(range(n_sensors))
    selected: list = []
    selected_ids = []
    errors = []

    for step in range(min(max_sensors, n_sensors)):
        best_idx = -1
        best_error = float("inf")

        # Need at least n_coils sensors for a full-rank system
        candidate_indices = list(available)

        for idx in candidate_indices:
            trial = selected + [idx]
            G_trial = G[trial, :]
            if len(trial) < n_coils:
                # Use pseudo-inverse based metric for underdetermined systems
                try:
                    s = np.linalg.svd(G_trial, compute_uv=False)
                    # Sum of inverse squared singular values (partial A-opt)
                    error = float(np.sum(1.0 / (s**2 + 1e-20)))
                except np.linalg.LinAlgError:
                    error = float("inf")
            else:
                error = reconstruction_error_metric(G_trial, noise_std)

            if error < best_error:
                best_error = error
                best_idx = idx

        if best_idx < 0:
            break

        selected.append(best_idx)
        selected_ids.append(sensor_ids[best_idx])
        available.remove(best_idx)
        errors.append(best_error)

    return selected_ids, errors


def sensor_type_analysis(
    G: np.ndarray,
    sensor_config: SensorConfig,
    noise_std: float = 1.0,
) -> dict[str, float]:
    """Compare reconstruction quality by sensor type.

    Args:
        G: Full Green's function matrix (flux loops first, then Mirnov).
        sensor_config: Sensor configuration.
        noise_std: Noise standard deviation.

    Returns:
        Dict with keys 'flux_only', 'mirnov_only', 'both' → error metric.
    """
    n_fl = sensor_config.n_flux_loops
    n_total = sensor_config.n_total

    G_fl = G[:n_fl, :]
    G_mp = G[n_fl:n_total, :]

    return {
        "flux_only": reconstruction_error_metric(G_fl, noise_std),
        "mirnov_only": reconstruction_error_metric(G_mp, noise_std),
        "both": reconstruction_error_metric(G, noise_std),
    }


def find_minimum_viable_set(
    G: np.ndarray,
    sensor_ids: list[str],
    coil_names: list[str],
    noise_std: float = 1.0,
    ip_threshold: float = 0.05,
    max_sensors: int | None = None,
) -> tuple[list[str], int]:
    """Find minimum sensor set meeting quality thresholds.

    Uses greedy forward selection and identifies the knee where
    adding more sensors gives diminishing returns.

    Args:
        G: Green's function matrix.
        sensor_ids: Sensor identifiers.
        coil_names: Coil names.
        noise_std: Noise standard deviation.
        ip_threshold: Max acceptable Ip relative error.
        max_sensors: Max sensors to consider.

    Returns:
        viable_ids: Minimum viable sensor set.
        n_viable: Number of sensors in viable set.
    """
    if max_sensors is None:
        max_sensors = len(sensor_ids)

    selected_ids, errors = greedy_forward_selection(
        G, sensor_ids, max_sensors=max_sensors, noise_std=noise_std,
    )

    n_coils = G.shape[1]

    # Find the knee: first point where error reduction < 1% of current error
    for k in range(n_coils, len(errors)):
        if k > 0:
            rel_improvement = (errors[k - 1] - errors[k]) / max(errors[k - 1], 1e-20)
            if rel_improvement < 0.01:
                return selected_ids[:k], k

    return selected_ids, len(selected_ids)
