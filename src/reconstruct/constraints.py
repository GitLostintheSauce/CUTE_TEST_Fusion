"""Measurement-to-constraint mapping for equilibrium reconstruction."""
from __future__ import annotations

import numpy as np

from src.forward.filament import loop_field
from src.forward.model import flux_loop, mirnov_probe
from src.forward.sensors import SensorConfig


def measurements_to_vector(
    measurements: dict[str, float], sensor_config: SensorConfig
) -> np.ndarray:
    """Convert a measurement dict (sensor_id -> value) to an ordered numpy vector.

    Order: all flux loops first (in sensor_config order), then all Mirnov probes.
    """
    vec = np.empty(sensor_config.n_total)
    for i, sensor in enumerate(sensor_config.flux_loops):
        vec[i] = measurements[sensor["id"]]
    offset = sensor_config.n_flux_loops
    for i, sensor in enumerate(sensor_config.mirnov_probes):
        vec[offset + i] = measurements[sensor["id"]]
    return vec


def forward_to_vector(mygs, sensor_config: SensorConfig) -> np.ndarray:
    """Evaluate the forward model for all sensors, returning an ordered vector."""
    psi_eval = mygs.get_field_eval("psi")
    B_eval = mygs.get_field_eval("B")

    vec = np.empty(sensor_config.n_total)
    for i, sensor in enumerate(sensor_config.flux_loops):
        vec[i] = flux_loop(psi_eval, (sensor["R"], sensor["Z"]))
    offset = sensor_config.n_flux_loops
    for i, sensor in enumerate(sensor_config.mirnov_probes):
        vec[offset + i] = mirnov_probe(B_eval, (sensor["R"], sensor["Z"]), sensor["angle"])
    return vec


def sensor_ids_ordered(sensor_config: SensorConfig) -> list[str]:
    """Return sensor IDs in the same order as the measurement vector."""
    ids = [s["id"] for s in sensor_config.flux_loops]
    ids.extend(s["id"] for s in sensor_config.mirnov_probes)
    return ids


def estimate_ip_from_mirnov(
    measurements: dict[str, float],
    sensor_config: SensorConfig,
    R0: float = 0.32,
    Z0: float = 0.0,
) -> float:
    """Rough plasma current from the magnetic probes, as a solver starting guess.

    Treats the plasma as one circular current filament at the nominal center
    (R0, Z0) and finds the current whose field best matches the probe readings
    in the least-squares sense: with g_i the reading probe i would give for
    1 A, Ip = sum(m_i g_i) / sum(g_i^2). This works for any probe layout.

    Limits: the probes also see the coil fields, and the real plasma is neither
    a thin filament nor exactly at (R0, Z0), so this is a starting value for the
    solver, not a measurement.
    """
    R = np.array([s["R"] for s in sensor_config.mirnov_probes])
    Z = np.array([s["Z"] for s in sensor_config.mirnov_probes])
    angle = np.array([s["angle"] for s in sensor_config.mirnov_probes])
    if R.size == 0:
        return 200.0e3  # no probes: fall back to nominal CUTE Ip

    m = np.array([measurements[s["id"]] for s in sensor_config.mirnov_probes])
    b_r, b_z = loop_field(R0, Z0, 1.0, R, Z)
    g = b_r * np.cos(angle) + b_z * np.sin(angle)
    return float(abs(m @ g) / (g @ g))
