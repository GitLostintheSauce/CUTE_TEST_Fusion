"""Sensor geometry configuration for CUTE diagnostics.

The positions come from ``config/cute_diagnostics.json``, a transcription of
the CUTE group's diagnostics table: 39 flux loops and 32 magnetic probe
entries. Every sensor sits about 0.5 cm outside the outer surface of the
vacuum vessel in ``config/CUTE_geom.json``, which is where sensors mounted on
the vessel wall would be, and ``tests/test_forward.py`` checks that.

This project models a 2D poloidal slice and assumes the plasma is the same all
the way around the torus (axisymmetry). Under that assumption a probe's reading
depends only on its (R, Z) position and its direction (N_R, N_Z), not on its
toroidal angle. So probe entries that repeat an earlier entry's position and
direction exactly would produce identical signals, and are left out:

- ``S2`` and ``S13`` at 0, 90, 180 and 270 degrees form a toroidal ring. On the
  real machine that ring detects non-axisymmetric plasma motion, which a 2D
  model cannot represent. Every entry in it repeats ``S02`` or ``S13``.

The ``S01@180`` to ``S12@180`` probes are kept. Their N_R has the opposite
sign to the probe at 0 degrees, so in the 2D model each pair measures two
different combinations of B_R and B_Z. Result: 39 flux loops and 25 probes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIAGNOSTICS_PATH = Path(__file__).resolve().parents[2] / "config" / "cute_diagnostics.json"


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


def _channel_id(name: str) -> str:
    """'S01@180' -> 'S01_180', so ids are plain identifiers in HDF5 and plots."""
    return name.replace("@", "_")


def generate_cute_sensors(path: Path | str = DIAGNOSTICS_PATH) -> SensorConfig:
    """Load the CUTE sensor layout from the diagnostics table.

    Each probe's direction (N_R, N_Z) becomes an angle in the poloidal plane,
    ``atan2(N_Z, N_R)``, so the probe reads ``B_R cos(angle) + B_Z sin(angle)``
    as :func:`src.forward.model.mirnov_probe` and the reduced model expect.
    """
    table = json.loads(Path(path).read_text())

    flux_loops = [
        {"id": fl["name"], "R": float(fl["R"]), "Z": float(fl["Z"])}
        for fl in table["flux_loops"]
    ]

    mirnov_probes = []
    seen: set[tuple[float, float, float, float]] = set()
    for p in table["magnetic_probes"]:
        key = (p["R"], p["Z"], p["N_R"], p["N_Z"])
        if key in seen:
            continue  # same reading as an earlier probe in a 2D model
        seen.add(key)
        mirnov_probes.append({
            "id": _channel_id(p["name"]),
            "R": float(p["R"]),
            "Z": float(p["Z"]),
            "angle": float(np.arctan2(p["N_Z"], p["N_R"])),
        })

    return SensorConfig(flux_loops=flux_loops, mirnov_probes=mirnov_probes)
