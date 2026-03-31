"""Sensor geometry configuration for CUTE diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorConfig:
    """Configuration for all diagnostic sensors."""

    flux_loops: list[dict]  # Each: {id, R, Z}
    mirnov_probes: list[dict]  # Each: {id, R, Z, angle} (angle in radians)

    @property
    def n_flux_loops(self) -> int:
        return len(self.flux_loops)

    @property
    def n_mirnov_probes(self) -> int:
        return len(self.mirnov_probes)

    @property
    def n_total(self) -> int:
        return self.n_flux_loops + self.n_mirnov_probes


def generate_cute_sensors() -> SensorConfig:
    """Generate sensor positions for CUTE based on vacuum vessel geometry.

    Places flux loops around the inner VV surface and Mirnov probes
    at toroidally-distributed locations on the outboard midplane and
    along the VV contour.
    """
    flux_loops = []
    mirnov_probes = []

    # Flux loops: distributed around the inner VV contour
    # CUTE VV inner contour spans roughly R=0.115-0.51, Z=-0.74 to 0.74
    # Place 56 flux loops at regular intervals around the poloidal cross-section

    # Inboard side (R ~ 0.14-0.16, various Z)
    n_inboard = 14
    z_inboard = np.linspace(-0.60, 0.60, n_inboard)
    for i, z in enumerate(z_inboard):
        flux_loops.append({"id": f"FL_IB{i+1:02d}", "R": 0.155, "Z": z})

    # Outboard midplane region (various R, Z near 0)
    n_outboard_mid = 10
    r_outboard = np.linspace(0.25, 0.50, n_outboard_mid)
    for i, r in enumerate(r_outboard):
        flux_loops.append({"id": f"FL_OB{i+1:02d}", "R": r, "Z": 0.0})

    # Upper outboard (R ~ 0.20-0.45, Z > 0)
    n_upper = 16
    angles_upper = np.linspace(0.2, 1.4, n_upper)  # poloidal angle
    for i, theta in enumerate(angles_upper):
        r = 0.32 + 0.18 * np.cos(theta)
        z = 0.20 * np.sin(theta) + 0.10
        flux_loops.append({"id": f"FL_UP{i+1:02d}", "R": max(r, 0.16), "Z": z})

    # Lower outboard (mirror of upper)
    for i, theta in enumerate(angles_upper):
        r = 0.32 + 0.18 * np.cos(theta)
        z = -(0.20 * np.sin(theta) + 0.10)
        flux_loops.append({"id": f"FL_LO{i+1:02d}", "R": max(r, 0.16), "Z": z})

    # Mirnov probes: measure local B-field components
    # Distributed around the VV inner surface

    # Outboard midplane probes (Br measurement)
    n_ob_mirnov = 20
    z_ob = np.linspace(-0.30, 0.30, n_ob_mirnov)
    for i, z in enumerate(z_ob):
        # Radial orientation (measuring Br)
        mirnov_probes.append({
            "id": f"MP_OBR{i+1:02d}", "R": 0.48, "Z": z, "angle": 0.0
        })

    # Outboard midplane probes (Bz measurement)
    n_ob_bz = 20
    z_ob_bz = np.linspace(-0.30, 0.30, n_ob_bz)
    for i, z in enumerate(z_ob_bz):
        # Vertical orientation (measuring Bz)
        mirnov_probes.append({
            "id": f"MP_OBZ{i+1:02d}", "R": 0.48, "Z": z, "angle": np.pi / 2
        })

    # Inboard probes (Bz measurement)
    n_ib_mirnov = 18
    z_ib = np.linspace(-0.50, 0.50, n_ib_mirnov)
    for i, z in enumerate(z_ib):
        mirnov_probes.append({
            "id": f"MP_IBZ{i+1:02d}", "R": 0.16, "Z": z, "angle": np.pi / 2
        })

    # Top/bottom probes
    n_tb = 16
    r_tb = np.linspace(0.20, 0.45, n_tb // 2)
    for i, r in enumerate(r_tb):
        mirnov_probes.append({
            "id": f"MP_TOP{i+1:02d}", "R": r, "Z": 0.35, "angle": 0.0
        })
        mirnov_probes.append({
            "id": f"MP_BOT{i+1:02d}", "R": r, "Z": -0.35, "angle": 0.0
        })

    return SensorConfig(flux_loops=flux_loops, mirnov_probes=mirnov_probes)
