"""Vacuum vessel eddy current compensation for CUTE reconstruction.

CUTE's vacuum vessel (η = 1.26×10⁻⁵ Ω·m) induces eddy currents during
transient events (current ramps, disruptions). These eddy currents produce
additional flux that sensors measure. This module models the VV impulse
response and subtracts the eddy contribution from measurements.

The VV response is modeled as a sum of exponential decays:

    H_vv(t) = Σ_k A_k * exp(-t / τ_k)

where τ_k are the wall eigenmode time constants and A_k are the amplitude
matrices (per sensor) for each mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from src.forward.model import flux_loop, mirnov_probe
from src.forward.sensors import SensorConfig


@dataclass
class VesselResponse:
    """Vessel eddy current response model.

    Attributes:
        time_constants: Decay time constants τ_k in seconds, shape (n_modes,).
        amplitudes: Amplitude matrix A, shape (n_sensors, n_modes). Each column
            gives the sensor response to mode k at t=0 for a unit step change.
        coil_name: Name of the coil used for the step response calibration.
    """
    time_constants: np.ndarray
    amplitudes: np.ndarray
    coil_name: str = "CS01"


def compute_vessel_response(
    mygs,
    sensor_config: SensorConfig,
    step_coil: str = "CS01",
    step_amplitude: float = 100.0,
    n_modes: int = 3,
    dt: float = 1e-5,
    n_steps: int = 80,
) -> VesselResponse:
    """Compute the VV eddy current response using a TD step simulation.

    Applies a step change in one coil current and records the transient
    eddy contribution at all sensor locations.

    Args:
        mygs: Configured TokaMaker instance.
        sensor_config: Sensor configuration.
        step_coil: Coil to apply the step change to.
        step_amplitude: Step change magnitude in Amperes.
        n_modes: Number of exponential modes to fit.
        dt: Time step for TD simulation in seconds.
        n_steps: Number of TD time steps.

    Returns:
        VesselResponse with fitted time constants and amplitudes.
    """
    coil_names = sorted(mygs.coil_sets.keys())

    def _eval_sensors():
        """Evaluate all sensors for current psi/B state."""
        psi_eval = mygs.get_field_eval("psi")
        B_eval = mygs.get_field_eval("B")
        vec = np.empty(sensor_config.n_total)
        for si, sensor in enumerate(sensor_config.flux_loops):
            vec[si] = flux_loop(psi_eval, (sensor["R"], sensor["Z"]))
        offset = sensor_config.n_flux_loops
        for si, sensor in enumerate(sensor_config.mirnov_probes):
            vec[offset + si] = mirnov_probe(
                B_eval, (sensor["R"], sensor["Z"]), sensor["angle"]
            )
        return vec

    # Compute steady-state BEFORE eig_wall (which corrupts vacuum solver)
    c_pre = {cn: 0.0 for cn in coil_names}
    c_pre[coil_names[0]] = 1e-6  # tiny baseline to avoid zero flux
    mygs.set_saddles(None)
    mygs.set_isoflux(None)

    c_post = dict(c_pre)
    c_post[step_coil] = step_amplitude
    mygs.set_coil_currents(c_post)
    mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
    mygs.solve(vacuum=True)
    y_ss = _eval_sensors()

    # Initial state for TD
    mygs.set_coil_currents(c_pre)
    mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
    mygs.solve(vacuum=True)

    # Get wall eigenvalues (NOTE: eig_wall corrupts vacuum solver state,
    # so all vacuum solves must happen before this call)
    eigenvalues, _ = mygs.eig_wall(neigs=n_modes, pm=False)
    eigen_rates = eigenvalues[:, 0]
    tau_eigen = 1.0 / eigen_rates

    # Setup TD solver and apply step change
    mygs.setup_td(dt, 1e-8, 1e-8)
    mygs.set_coil_currents(c_post)

    # Run TD simulation and record sensor responses
    times = []
    responses = []
    for i in range(n_steps):
        t = (i + 1) * dt
        mygs.step_td(t, dt)
        responses.append(_eval_sensors())
        times.append(t)

    times_arr = np.array(times)
    responses_arr = np.array(responses)  # (n_steps, n_sensors)

    # Eddy contribution = TD response - steady state
    eddy = responses_arr - y_ss[np.newaxis, :]  # (n_steps, n_sensors)

    # Fit multi-exponential to each sensor's eddy response
    # H(t) = Σ_k A_k * exp(-t/τ_k) where τ_k are from eig_wall
    # Linear least-squares for A_k given fixed τ_k
    tau_fit = tau_eigen[:n_modes]
    basis = np.exp(-times_arr[:, np.newaxis] / tau_fit[np.newaxis, :])  # (n_steps, n_modes)

    # Solve for amplitudes: eddy ≈ basis @ A.T
    # A.T shape (n_modes, n_sensors), solve basis @ A.T = eddy
    A_T, _, _, _ = np.linalg.lstsq(basis, eddy, rcond=None)
    amplitudes = A_T.T  # (n_sensors, n_modes)

    # Normalize amplitudes per unit step
    amplitudes /= step_amplitude

    return VesselResponse(
        time_constants=tau_fit,
        amplitudes=amplitudes,
        coil_name=step_coil,
    )


def compensate_eddy(
    measurements_timeseries: np.ndarray,
    times: np.ndarray,
    coil_currents_timeseries: np.ndarray,
    vessel_response: VesselResponse,
) -> np.ndarray:
    """Subtract eddy current contribution from measurement timeseries.

    Args:
        measurements_timeseries: Sensor values, shape (n_times, n_sensors).
        times: Time values in seconds, shape (n_times,).
        coil_currents_timeseries: Coil current values, shape (n_times, n_coils).
            Only the column corresponding to vessel_response.coil_name is used
            (the response is calibrated for that coil).
        vessel_response: Fitted vessel response model.

    Returns:
        Compensated measurements, shape (n_times, n_sensors).
    """
    n_times = len(times)
    n_sensors = measurements_timeseries.shape[1]
    tau = vessel_response.time_constants
    A = vessel_response.amplitudes  # (n_sensors, n_modes)

    # Compute dI/dt from coil currents (finite difference)
    dt_arr = np.diff(times, prepend=times[0] - (times[1] - times[0]))
    dI_dt = (
        np.diff(coil_currents_timeseries, axis=0, prepend=coil_currents_timeseries[:1])
        / dt_arr[:, np.newaxis]
    )

    # Sum over all coils (simplified: use total dI/dt as proxy)
    # For a more accurate model, we'd need per-coil response matrices.
    # Here we sum all coil current changes as a single driver.
    dI_total = np.sum(dI_dt, axis=1)  # (n_times,)

    # Compute eddy contribution via discrete convolution
    y_eddy = np.zeros((n_times, n_sensors))
    for t_idx in range(1, n_times):
        for k in range(len(tau)):
            # Contribution from each past time step, decayed
            for t_past in range(t_idx):
                elapsed = times[t_idx] - times[t_past]
                dt_past = dt_arr[t_past]
                impulse = dI_total[t_past] * dt_past
                y_eddy[t_idx] += A[:, k] * impulse * np.exp(-elapsed / tau[k])

    return measurements_timeseries - y_eddy


def compensate_eddy_fast(
    measurements_timeseries: np.ndarray,
    times: np.ndarray,
    coil_currents_timeseries: np.ndarray,
    vessel_response: VesselResponse,
) -> np.ndarray:
    """Fast eddy compensation using recursive exponential filter.

    For each mode k with time constant τ_k, the eddy state s_k evolves as:
        s_k[n] = α_k * s_k[n-1] + (1-α_k) * dI[n]
    where α_k = exp(-dt/τ_k).

    Args:
        measurements_timeseries: Sensor values, shape (n_times, n_sensors).
        times: Time values in seconds, shape (n_times,).
        coil_currents_timeseries: Coil current values, shape (n_times, n_coils).
        vessel_response: Fitted vessel response model.

    Returns:
        Compensated measurements, shape (n_times, n_sensors).
    """
    n_times = len(times)
    tau = vessel_response.time_constants
    A = vessel_response.amplitudes  # (n_sensors, n_modes)
    n_modes = len(tau)

    # Total coil current change
    dI = np.diff(coil_currents_timeseries.sum(axis=1), prepend=0.0)

    # Recursive filter for each mode
    y_eddy = np.zeros_like(measurements_timeseries)
    state = np.zeros(n_modes)  # mode states

    for t_idx in range(1, n_times):
        dt_step = times[t_idx] - times[t_idx - 1]
        for k in range(n_modes):
            alpha = np.exp(-dt_step / tau[k])
            state[k] = alpha * state[k] + dI[t_idx]
            y_eddy[t_idx] += A[:, k] * state[k]

    return measurements_timeseries - y_eddy


def save_vessel_response(path: str | Path, vr: VesselResponse) -> None:
    """Save vessel response model to HDF5."""
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("time_constants", data=vr.time_constants)
        f.create_dataset("amplitudes", data=vr.amplitudes)
        f.attrs["coil_name"] = vr.coil_name


def load_vessel_response(path: str | Path) -> VesselResponse:
    """Load vessel response model from HDF5."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        return VesselResponse(
            time_constants=f["time_constants"][:],
            amplitudes=f["amplitudes"][:],
            coil_name=str(f.attrs["coil_name"]),
        )


def fit_quality(
    times: np.ndarray,
    eddy_actual: np.ndarray,
    vessel_response: VesselResponse,
    step_amplitude: float = 1.0,
) -> np.ndarray:
    """Compute R² of exponential fit for each sensor.

    Args:
        times: Time values in seconds.
        eddy_actual: Actual eddy contribution, shape (n_times, n_sensors).
        vessel_response: Fitted model.
        step_amplitude: Step amplitude used to generate eddy_actual.

    Returns:
        r_squared: R² values, shape (n_sensors,).
    """
    tau = vessel_response.time_constants
    A = vessel_response.amplitudes * step_amplitude
    basis = np.exp(-times[:, np.newaxis] / tau[np.newaxis, :])
    eddy_pred = basis @ A.T  # (n_times, n_sensors)

    ss_res = np.sum((eddy_actual - eddy_pred) ** 2, axis=0)
    ss_tot = np.sum((eddy_actual - eddy_actual.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-30)
    return r2
