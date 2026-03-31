"""Forward model: compute synthetic sensor signals from equilibria."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sensors import SensorConfig


def flux_loop(psi_eval, sensor_pos: tuple[float, float]) -> float:
    """Evaluate poloidal flux at a sensor (R, Z) location.

    Args:
        psi_eval: TokaMaker field evaluator for "psi".
        sensor_pos: (R, Z) in meters.

    Returns:
        Poloidal flux in Wb.
    """
    pt = np.array(sensor_pos, dtype=np.float64)
    val = psi_eval.eval(pt)
    return float(val[0])


def mirnov_probe(
    B_eval, sensor_pos: tuple[float, float], sensor_angle: float
) -> float:
    """Evaluate B-field component at a probe location.

    Args:
        B_eval: TokaMaker field evaluator for "B".
        sensor_pos: (R, Z) in meters.
        sensor_angle: Orientation angle in radians (0=radial/Br, pi/2=vertical/Bz).

    Returns:
        B-field component in Tesla along the probe orientation.
    """
    pt = np.array(sensor_pos, dtype=np.float64)
    B_vals = B_eval.eval(pt)  # [Br, Bt, Bz]
    # Project onto probe orientation in poloidal plane
    return float(B_vals[0] * np.cos(sensor_angle) + B_vals[2] * np.sin(sensor_angle))


def full_diagnostic_set(
    mygs,
    sensor_config: SensorConfig,
    time_index: float = 0.0,
) -> pd.DataFrame:
    """Evaluate all sensors for a single equilibrium state.

    Args:
        mygs: TokaMaker instance with a solved equilibrium.
        sensor_config: Sensor configuration.
        time_index: Time label for this measurement.

    Returns:
        DataFrame with one row and columns for each sensor.
    """
    psi_eval = mygs.get_field_eval("psi")
    B_eval = mygs.get_field_eval("B")

    data = {"time": time_index}

    for sensor in sensor_config.flux_loops:
        val = flux_loop(psi_eval, (sensor["R"], sensor["Z"]))
        data[sensor["id"]] = val

    for sensor in sensor_config.mirnov_probes:
        val = mirnov_probe(B_eval, (sensor["R"], sensor["Z"]), sensor["angle"])
        data[sensor["id"]] = val

    return pd.DataFrame([data])


def full_diagnostic_timeseries(
    mygs,
    sensor_config: SensorConfig,
    psi_timeseries: list[np.ndarray],
    times: list[float],
) -> pd.DataFrame:
    """Evaluate all sensors across a time series of equilibria.

    Args:
        mygs: TokaMaker instance.
        sensor_config: Sensor configuration.
        psi_timeseries: List of psi arrays, one per time step.
        times: Corresponding time values.

    Returns:
        DataFrame with one row per time step and columns for each sensor.
    """
    frames = []
    for t, psi in zip(times, psi_timeseries):
        mygs.set_psi(psi)
        frame = full_diagnostic_set(mygs, sensor_config, time_index=t)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
