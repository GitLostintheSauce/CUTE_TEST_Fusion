"""Reduced-physics forward model and labeled-dataset generation.

We model the CUTE plasma as a small disk of circular current filaments and
evaluate the magnetic diagnostics (flux loops and Mirnov probes) analytically
via :mod:`src.ml.physics`. Sampling many plasma states and computing their
sensor signatures yields a labeled dataset:

    X  sensor signals   (n_samples, n_sensors=130)
    y  plasma parameters (n_samples, 4) = [Ip, R0, Z0, a]

This is a *reduced* forward model (rigid current disk, no free-boundary
Grad-Shafranov solve), used to train and benchmark the ML surrogate in a
fully self-contained way. It is labeled as reduced wherever it is surfaced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.forward.sensors import SensorConfig, generate_cute_sensors
from src.ml.physics import loop_field, loop_flux

# Plasma parameter names, in the fixed order used by X/y arrays.
PARAM_NAMES = ["Ip", "R0", "Z0", "a"]
PARAM_UNITS = ["A", "m", "m", "m"]

# Physically motivated sampling ranges for a CUTE-scale spherical tokamak.
PARAM_RANGES = {
    "Ip": (2.0e4, 2.5e5),   # plasma current [A]
    "R0": (0.28, 0.36),     # major radius [m]
    "Z0": (-0.06, 0.06),    # vertical position [m]
    "a": (0.05, 0.18),      # minor radius [m]
}


@dataclass
class SensorLayout:
    """Precomputed sensor coordinates for fast, vectorized forward evaluation."""

    fl_R: np.ndarray       # flux-loop R positions
    fl_Z: np.ndarray       # flux-loop Z positions
    mp_R: np.ndarray       # Mirnov R positions
    mp_Z: np.ndarray       # Mirnov Z positions
    mp_cos: np.ndarray     # cos(orientation) for each Mirnov probe
    mp_sin: np.ndarray     # sin(orientation) for each Mirnov probe

    @property
    def n_sensors(self) -> int:
        return len(self.fl_R) + len(self.mp_R)

    @classmethod
    def from_config(cls, config: SensorConfig | None = None) -> "SensorLayout":
        config = config or generate_cute_sensors()
        fl_R = np.array([s["R"] for s in config.flux_loops], dtype=float)
        fl_Z = np.array([s["Z"] for s in config.flux_loops], dtype=float)
        mp_R = np.array([s["R"] for s in config.mirnov_probes], dtype=float)
        mp_Z = np.array([s["Z"] for s in config.mirnov_probes], dtype=float)
        mp_ang = np.array([s["angle"] for s in config.mirnov_probes], dtype=float)
        return cls(fl_R, fl_Z, mp_R, mp_Z, np.cos(mp_ang), np.sin(mp_ang))


def _plasma_filaments(Ip, R0, Z0, a, n_ring: int = 6):
    """Represent a plasma as a central filament plus a ring, sharing total current Ip.

    Giving the plasma finite extent (radius ``a``) is what makes the minor
    radius observable in the sensor signals, so the surrogate can learn it.
    """
    currents = [Ip / (n_ring + 1)]
    a_fil = [R0]
    z_fil = [Z0]
    for k in range(n_ring):
        theta = 2.0 * np.pi * k / n_ring
        a_fil.append(R0 + 0.6 * a * np.cos(theta))
        z_fil.append(Z0 + 0.6 * a * np.sin(theta))
        currents.append(Ip / (n_ring + 1))
    return np.array(a_fil), np.array(z_fil), np.array(currents)


def forward_signals(Ip, R0, Z0, a, layout: SensorLayout) -> np.ndarray:
    """Compute the 130-element sensor vector for one plasma state.

    Flux loops measure poloidal flux; Mirnov probes measure the B component
    along their orientation. Returns a 1-D array [flux_loops..., mirnov...].
    """
    a_fil, z_fil, curr = _plasma_filaments(Ip, R0, Z0, a)

    fl = np.zeros_like(layout.fl_R)
    mp_br = np.zeros_like(layout.mp_R)
    mp_bz = np.zeros_like(layout.mp_R)
    for af, zf, cf in zip(a_fil, z_fil, curr):
        fl += loop_flux(af, zf, cf, layout.fl_R, layout.fl_Z)
        br, bz = loop_field(af, zf, cf, layout.mp_R, layout.mp_Z)
        mp_br += br
        mp_bz += bz
    mp = mp_br * layout.mp_cos + mp_bz * layout.mp_sin
    return np.concatenate([fl, mp])


def generate_dataset(
    n_samples: int = 4000,
    noise_frac: float = 0.02,
    seed: int = 0,
    layout: SensorLayout | None = None,
) -> tuple[np.ndarray, np.ndarray, SensorLayout]:
    """Generate a labeled (X, y) dataset from the reduced forward model.

    Args:
        n_samples: Number of plasma states to sample.
        noise_frac: Gaussian sensor noise as a fraction of the per-channel
            signal scale (0 disables noise). Models diagnostic measurement error.
        seed: RNG seed for reproducibility.
        layout: Optional precomputed sensor layout.

    Returns:
        X: (n_samples, n_sensors) sensor signals.
        y: (n_samples, 4) plasma parameters [Ip, R0, Z0, a].
        layout: The SensorLayout used (for reuse at inference time).
    """
    rng = np.random.default_rng(seed)
    layout = layout or SensorLayout.from_config()

    y = np.empty((n_samples, len(PARAM_NAMES)))
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        y[:, j] = rng.uniform(lo, hi, n_samples)

    X = np.empty((n_samples, layout.n_sensors))
    for i in range(n_samples):
        X[i] = forward_signals(y[i, 0], y[i, 1], y[i, 2], y[i, 3], layout)

    if noise_frac > 0:
        # Per-channel scale from the dataset itself, plus a small absolute floor.
        scale = np.abs(X).mean(axis=0)
        scale = np.maximum(scale, 1e-9)
        X = X + rng.normal(0.0, 1.0, X.shape) * (noise_frac * scale)

    return X, y, layout
