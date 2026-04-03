"""Green's function matrix computation for EFIT-style reconstruction.

The Green's function matrix G relates coil currents to sensor measurements
in the vacuum (no plasma) limit:

    y_vacuum = G @ I_coils

where y_vacuum is a vector of sensor values (130 for CUTE: 56 flux loops +
74 Mirnov probes) and I_coils is a vector of coil currents (28 for CUTE).

G is computed by running TokaMaker with unit current in each coil (all others
zero, no plasma) and evaluating the field at all sensor locations.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from src.forward.model import flux_loop, mirnov_probe
from src.forward.sensors import SensorConfig


def compute_greens_matrix(
    mygs,
    sensor_config: SensorConfig,
    coil_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute the Green's function matrix by vacuum solves with unit coil currents.

    For each coil, sets unit current (all others zero), solves the vacuum field
    (Ip=0), and evaluates psi/B at all sensor locations.

    Args:
        mygs: Configured TokaMaker instance (mesh, regions set up, but profiles
              and targets will be overridden).
        sensor_config: Sensor configuration.
        coil_names: Ordered list of coil names. If None, uses sorted(mygs.coil_sets).

    Returns:
        G: Green's function matrix, shape (n_sensors, n_coils).
        coil_names: Ordered coil names corresponding to columns of G.
    """
    if coil_names is None:
        coil_names = sorted(mygs.coil_sets.keys())

    n_sensors = sensor_config.n_total
    n_coils = len(coil_names)
    G = np.zeros((n_sensors, n_coils))

    # Clear shape constraints for vacuum solve
    mygs.set_saddles(None)
    mygs.set_isoflux(None)

    def _vacuum_response(current_dict):
        """Solve vacuum field and return sensor response vector."""
        mygs.set_coil_currents(current_dict)
        mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
        mygs.solve(vacuum=True)
        psi_eval = mygs.get_field_eval("psi")
        B_eval = mygs.get_field_eval("B")
        vec = np.empty(n_sensors)
        for i, sensor in enumerate(sensor_config.flux_loops):
            vec[i] = flux_loop(psi_eval, (sensor["R"], sensor["Z"]))
        offset = sensor_config.n_flux_loops
        for i, sensor in enumerate(sensor_config.mirnov_probes):
            vec[offset + i] = mirnov_probe(
                B_eval, (sensor["R"], sensor["Z"]), sensor["angle"]
            )
        return vec

    # Compute baseline response (tiny current to avoid zero-flux error).
    # init_psi adds a background flux that must be subtracted to isolate
    # the linear coil contribution.
    baseline_currents = {cn: 0.0 for cn in coil_names}
    baseline_currents[coil_names[0]] = 1e-6
    y_baseline = _vacuum_response(baseline_currents)

    for j, name in enumerate(coil_names):
        # Set unit current in this coil (plus baseline)
        currents = dict(baseline_currents)
        currents[name] = currents.get(name, 0.0) + 1.0
        y_j = _vacuum_response(currents)
        G[:, j] = y_j - y_baseline

    return G, coil_names


def compute_plasma_response(
    mygs,
    sensor_config: SensorConfig,
) -> np.ndarray:
    """Compute the plasma contribution to sensor measurements.

    Runs the reference equilibrium solve and subtracts the vacuum coil
    contribution (using the current coil currents and Green's matrix)
    to isolate the plasma-only response.

    This is called after a solve() to extract y_plasma = y_total - G @ I_coils.

    Args:
        mygs: TokaMaker with a solved equilibrium.
        sensor_config: Sensor configuration.

    Returns:
        y_total: Total sensor values from the current equilibrium.
    """
    from src.reconstruct.constraints import forward_to_vector

    return forward_to_vector(mygs, sensor_config)


def save_greens_matrix(
    path: str | Path,
    G: np.ndarray,
    coil_names: list[str],
) -> None:
    """Save Green's function matrix to HDF5."""
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("G", data=G)
        f.attrs["coil_names"] = coil_names


def load_greens_matrix(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load Green's function matrix from HDF5."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        G = f["G"][:]
        coil_names = list(f.attrs["coil_names"])
    return G, coil_names
