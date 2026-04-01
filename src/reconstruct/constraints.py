"""Measurement-to-constraint mapping for equilibrium reconstruction."""
from __future__ import annotations

import numpy as np

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
    measurements: dict[str, float], sensor_config: SensorConfig
) -> float:
    """Estimate plasma current from Mirnov probe measurements.

    Uses a simplified Ampere's law: Ip ~ (2*pi*R_avg / mu0) * <B_pol>
    where <B_pol> is the average poloidal field from outboard probes.
    """
    mu0 = 4.0e-7 * np.pi

    # Use outboard Bz probes (they measure the dominant poloidal field component)
    bz_values = []
    for sensor in sensor_config.mirnov_probes:
        if sensor["id"].startswith("MP_OBZ"):
            bz_values.append(measurements[sensor["id"]])

    if not bz_values:
        return 200.0e3  # fallback to nominal CUTE Ip

    avg_bz = np.mean(bz_values)
    R_avg = 0.48  # outboard probe radius
    # Ampere's law: B_z ~ mu0 * Ip / (2*pi*R) for a simple current loop
    ip_est = abs(avg_bz) * 2.0 * np.pi * R_avg / mu0
    return ip_est
