"""Analytic magnetics for a circular current filament.

The poloidal flux and magnetic field of a single circular loop of radius ``a``
centered on the machine axis at height ``zc``, carrying current ``I``, have
closed-form expressions in terms of the complete elliptic integrals K and E.
These are the standard results (see Simpson et al., NASA/TM-2013-217918, and
Jackson, Classical Electrodynamics, sec. 5.5).

SciPy's ``ellipk``/``ellipe`` take the parameter ``m = k**2`` (not the modulus
``k``); we follow that convention throughout.

We model a plasma as a small set of such filaments, which lets us build a
physically grounded (if reduced) forward model for the CUTE magnetic
diagnostics without a full free-boundary Grad-Shafranov solve. The
implementation is validated against a direct Biot-Savart quadrature in the
test suite.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk

MU0 = 4.0e-7 * np.pi  # vacuum permeability [T*m/A]


def _k_squared(a, R, dZ):
    """m = k**2 = 4 a R / ((a + R)**2 + dZ**2), clipped away from 1.0."""
    denom = (a + R) ** 2 + dZ ** 2
    m = 4.0 * a * R / np.where(denom == 0.0, np.finfo(float).tiny, denom)
    # Keep strictly inside [0, 1) so the elliptic integrals stay finite.
    return np.clip(m, 0.0, 1.0 - 1e-12)


def loop_flux(a: float, zc: float, current: float, R, Z):
    """Poloidal flux psi = 2*pi*R*A_phi from a circular filament.

    Args:
        a: Loop radius [m] (a > 0).
        zc: Loop height [m].
        current: Loop current [A].
        R, Z: Evaluation coordinates [m] (scalars or broadcastable arrays).

    Returns:
        Poloidal flux [Wb], same shape as the broadcast of R and Z.
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    dZ = Z - zc
    Rsafe = np.where(R == 0.0, np.finfo(float).tiny, R)
    m = _k_squared(a, Rsafe, dZ)
    # A_phi = (mu0 I / pi) * sqrt(a / R) / sqrt(m) * ((1 - m/2) K - E)
    a_phi = (MU0 * current / np.pi) * np.sqrt(a / Rsafe) / np.sqrt(m) * (
        (1.0 - 0.5 * m) * ellipk(m) - ellipe(m)
    )
    psi = 2.0 * np.pi * R * a_phi
    return psi


def loop_field(a: float, zc: float, current: float, R, Z):
    """Magnetic field components (B_R, B_Z) from a circular filament.

    Args:
        a: Loop radius [m] (a > 0).
        zc: Loop height [m].
        current: Loop current [A].
        R, Z: Evaluation coordinates [m].

    Returns:
        (B_R, B_Z) in Tesla, each the shape of the broadcast of R and Z.
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    dZ = Z - zc
    Rsafe = np.where(R == 0.0, np.finfo(float).tiny, R)

    sum_sq = (a + Rsafe) ** 2 + dZ ** 2
    diff_sq = (a - Rsafe) ** 2 + dZ ** 2
    m = _k_squared(a, Rsafe, dZ)
    K = ellipk(m)
    E = ellipe(m)
    pref = MU0 * current / (2.0 * np.pi)
    sqrt_sum = np.sqrt(sum_sq)

    b_r = pref * (dZ / (Rsafe * sqrt_sum)) * (
        -K + (a ** 2 + Rsafe ** 2 + dZ ** 2) / diff_sq * E
    )
    b_z = pref * (1.0 / sqrt_sum) * (
        K + (a ** 2 - Rsafe ** 2 - dZ ** 2) / diff_sq * E
    )
    return b_r, b_z


def biot_savart_field(a: float, zc: float, current: float, R: float, Z: float,
                      n_seg: int = 4000):
    """Direct Biot-Savart quadrature for one loop, used to validate the analytic form.

    Integrates dB over the circular loop numerically. Returns (B_R, B_Z) at a
    single evaluation point (R, Z) in the phi = 0 plane.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    dphi = 2.0 * np.pi / n_seg
    # Source points on the loop.
    sx = a * np.cos(phi)
    sy = a * np.sin(phi)
    sz = np.full_like(phi, zc)
    # Current-element direction (tangent) * |dl|.
    dlx = -a * np.sin(phi) * dphi
    dly = a * np.cos(phi) * dphi
    dlz = np.zeros_like(phi)
    # Field point in the phi = 0 plane: (R, 0, Z).
    rx, ry, rz = R - sx, 0.0 - sy, Z - sz
    dist = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    inv = 1.0 / dist ** 3
    # dB = (mu0 I / 4pi) dl x r / |r|^3
    # Only the in-plane (x) and vertical (z) components are needed; the
    # azimuthal (y) component integrates to ~0 in the phi = 0 plane.
    cx = dly * rz - dlz * ry
    cz = dlx * ry - dly * rx
    pref = MU0 * current / (4.0 * np.pi)
    bx = pref * np.sum(cx * inv)
    bz = pref * np.sum(cz * inv)
    # In the phi=0 plane, B_R is the x-component; B_phi (y) integrates to ~0.
    return float(bx), float(bz)
