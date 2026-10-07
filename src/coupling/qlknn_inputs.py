"""Turn a CUTE equilibrium into the ten inputs QLKNN_7_11 expects.

The big picture
---------------
An equilibrium (what TokaMaker computes) says what *shape* the plasma and its
magnetic field have at one instant. A transport model says how fast heat and
particles *leak* out through that field, which is what sets how hot the plasma
can get. QLKNN_7_11 is a neural network trained to imitate QuaLiKiz, a physics
code that computes the turbulent part of that leak. It does not take an
equilibrium file. It takes ten local numbers at each radius, and this module
computes them.

Where each input comes from
---------------------------
Four inputs come from the equilibrium, so from this repo:

* ``q``     safety factor: how many times a field line goes around the long
            way per trip around the short way.
* ``smag``  magnetic shear, ``(r/q) dq/dr``: how fast q changes with radius.
* ``x``     normalized radius ``r/a``, rescaled for aspect ratio (see below).
* the major radius ``R``, used to normalize the gradients.

The other six need temperature and density profiles. **Magnetic sensors do
not measure temperature or density,** so in this proof of concept they come
from :class:`KineticAssumptions`, which are placeholders and labeled as such:

* ``Ati``, ``Ate``  ``-R d(ln T)/dr``: how steeply ion and electron
  temperature fall with radius, in units of the major radius. Steep
  temperature gradients are what drive the turbulence.
* ``Ane``, ``Ani``  the same for electron and ion density.
* ``Ti_Te``         ion over electron temperature.
* ``LogNuStar``     log10 of collisionality: how often electrons collide
  compared with how often they bounce along the field. Cold, dense plasmas are
  very collisional.
* ``normni``        ion density over electron density (1 for a pure plasma).

Every formula follows TORAX v1.4.3 (``torax/_src/transport_model/
qualikiz_based_transport_model.py`` and ``qlknn_transport_model.py``,
``torax/_src/physics/collisions.py``), so the numbers here are what TORAX
would hand QLKNN for the same profiles.

The aspect-ratio rescaling
--------------------------
QLKNN's training data (QLKNN11D) was generated for one fixed plasma
proportion: minor radius over major radius ``a/R = 1/3``. Many quantities
that matter for turbulence, notably the fraction of electrons trapped on the
outer side of the torus, depend on the *local* inverse aspect ratio ``r/R``,
not on ``r/a``. TORAX therefore rescales ``x`` so that the local ``r/R`` seen
by the network matches the real plasma::

    x_qlknn = (r / a) * (a / R) / (1/3)

CUTE is much fatter than 1/3 (``a/R`` near 0.5), so ``x_qlknn`` passes the
training maximum of 0.95 well inside the plasma. That is one of the roadblocks
this proof of concept exists to surface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Physical constants (SI), the same values TORAX uses.
Q_E = 1.602176634e-19        # elementary charge [C]
M_E = 9.1093837e-31          # electron mass [kg]
M_AMU = 1.66053906660e-27    # atomic mass unit [kg]
EPS_0 = 8.8541878128e-12     # vacuum permittivity [F/m]
MU_0 = 4e-7 * np.pi          # vacuum permeability [H/m]
KEV_TO_J = 1e3 * Q_E

# QLKNN_7_11 input order (from the model's own config) and the fixed inverse
# aspect ratio of its training set (TORAX: _EPSILON_NN in
# qlknn_transport_model.py).
INPUT_NAMES = ["Ati", "Ate", "Ane", "Ani", "q", "smag", "x", "Ti_Te",
               "LogNuStar", "normni"]
EPSILON_NN = 1.0 / 3.0

# The two datasets QLKNN_7_11 was trained on, with the min and max of each
# input in each, from the model card that ships with fusion_surrogates
# (fusion_surrogates/qlknn/models/README.md, v0.4.7). The model's built-in
# ``inputs_and_ranges`` is the union of these two boxes, which is looser than
# the data: a point can sit inside the union yet far from any training sample
# (for example x = 0.5 with Ati = 100). So this module checks each box on its
# own.
TRAINING_BOXES: dict[str, dict[str, tuple[float, float]]] = {
    "QLKNN11D (core)": {
        "Ati": (0.0, 16.0), "Ate": (0.0, 16.0), "Ane": (-5.0, 5.0),
        "Ani": (-15.0, 15.0), "q": (0.66, 10.0), "smag": (-1.0, 4.0),
        "x": (0.1, 0.95), "Ti_Te": (0.25, 2.5), "LogNuStar": (-5.0, 0.0),
        "normni": (0.7, 1.0),
    },
    "QLKNN7D (edge)": {
        "Ati": (5.0, 150.0), "Ate": (5.0, 150.0), "Ane": (2.0, 110.0),
        "Ani": (2.0, 110.0), "q": (2.0, 30.0), "smag": (1.0, 40.0),
        "x": (0.95, 0.95), "Ti_Te": (1.0, 1.0), "LogNuStar": (-1.0, 0.477),
        "normni": (0.5, 1.0),
    },
}


@dataclass(frozen=True)
class FluxSurfaceGeometry:
    """Per-flux-surface quantities from a solved equilibrium.

    One entry per flux surface, ordered from the magnetic axis outward.
    ``r_mid`` is the midplane half-width of each surface, ``(R_out - R_in)/2``,
    which is the radius QuaLiKiz-style models use.

    Args:
        psi_norm: Normalized poloidal flux, 0 on axis and 1 at the boundary.
        q: Safety factor on each surface.
        R_in: Innermost major radius of each surface [m].
        R_out: Outermost major radius of each surface [m].
        R_major: Geometric major radius of the boundary [m].
        a_minor: Minor radius of the boundary [m].
        B0: Vacuum toroidal field at ``R_major`` [T].
    """

    psi_norm: np.ndarray
    q: np.ndarray
    R_in: np.ndarray
    R_out: np.ndarray
    R_major: float
    a_minor: float
    B0: float

    @property
    def r_mid(self) -> np.ndarray:
        return 0.5 * (self.R_out - self.R_in)

    @property
    def R_mid(self) -> np.ndarray:
        """Major radius of each surface's center, used in the bounce time."""
        return 0.5 * (self.R_out + self.R_in)

    @property
    def epsilon(self) -> np.ndarray:
        """Local inverse aspect ratio of each surface."""
        return (self.R_out - self.R_in) / (self.R_out + self.R_in)

    @property
    def rho(self) -> np.ndarray:
        """Normalized radius r/a, the coordinate the kinetic profiles use."""
        return self.r_mid / self.a_minor

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict) -> FluxSurfaceGeometry:
        def arr(key: str) -> np.ndarray:
            return np.asarray(d[key], dtype=float)

        return cls(psi_norm=arr("psi_norm"), q=arr("q"), R_in=arr("R_in"),
                   R_out=arr("R_out"), R_major=float(d["R_major"]),
                   a_minor=float(d["a_minor"]), B0=float(d["B0"]))


@dataclass(frozen=True)
class KineticAssumptions:
    """ASSUMED temperature and density profiles. Not measured, not CUTE data.

    The magnetic diagnostics this repo models cannot see temperature or
    density, so QLKNN's six kinetic inputs have to come from somewhere else.
    On a real machine that would be Thomson scattering, interferometry, or a
    transport simulation. Here they are placeholder shapes,

        value(rho) = edge + (core - edge) * (1 - rho**2) ** peaking,

    with defaults chosen to be plausible for a small, ohmically heated
    tokamak, not taken from CUTE. Every number QLKNN produces in this proof of
    concept inherits these guesses.

    Args:
        Te0_keV: Electron temperature on axis [keV].
        Te_edge_keV: Electron temperature at the boundary [keV].
        Ti_over_Te: Ion over electron temperature, held constant.
        ne0: Electron density on axis [m^-3].
        ne_edge: Electron density at the boundary [m^-3].
        T_peaking: Exponent setting how peaked the temperature profile is.
        n_peaking: Exponent setting how peaked the density profile is.
        A_i: Main ion mass [amu]. 1 for hydrogen, 2 for deuterium.
        Z_eff: Effective charge. 1 means a pure plasma; above 1, the
            impurity is taken to be carbon.
    """

    Te0_keV: float = 0.2
    Te_edge_keV: float = 0.02
    Ti_over_Te: float = 0.5
    ne0: float = 1.0e19
    ne_edge: float = 2.0e18
    T_peaking: float = 1.5
    n_peaking: float = 0.5
    A_i: float = 1.0
    Z_eff: float = 1.0


def kinetic_profiles(rho: np.ndarray, k: KineticAssumptions
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the assumed Te, Ti [keV] and ne [m^-3] at normalized radius rho."""
    rho = np.clip(np.asarray(rho, dtype=float), 0.0, 1.0)
    shape_T = (1.0 - rho**2) ** k.T_peaking
    shape_n = (1.0 - rho**2) ** k.n_peaking
    Te = k.Te_edge_keV + (k.Te0_keV - k.Te_edge_keV) * shape_T
    ne = k.ne_edge + (k.ne0 - k.ne_edge) * shape_n
    return Te, k.Ti_over_Te * Te, ne


def log_nu_star(Te_keV: np.ndarray, ne: np.ndarray, q: np.ndarray,
                R_mid: np.ndarray, epsilon: np.ndarray, Z_eff: float) -> np.ndarray:
    """log10 of electron collisionality, as TORAX computes it.

    Collision frequency (Wesson, 3rd ed., p. 727 and 729) divided by the
    bounce frequency of a trapped electron, ``sqrt(eps**3) v_te / (q R)``.
    Follows ``torax/_src/physics/collisions.py::calc_nu_star``.
    """
    log_lambda = 31.3 - 0.5 * np.log(ne) + np.log(Te_keV * 1e3)
    log_tau_e = (np.log(12 * np.pi**1.5 / (ne * log_lambda))
                 - 4 * np.log(Q_E) + 0.5 * np.log(M_E / 2.0)
                 + 2 * np.log(EPS_0) + 1.5 * np.log(Te_keV * KEV_TO_J))
    nu_e = Z_eff / np.exp(log_tau_e)
    tau_bounce = q * R_mid / (np.clip(epsilon, 1e-9, None) ** 1.5
                              * np.sqrt(Te_keV * KEV_TO_J / M_E))
    return np.log10(nu_e * tau_bounce)


def gyrobohm_diffusivity(Ti_keV: np.ndarray, B0: float, A_i: float,
                         a_minor: float) -> np.ndarray:
    """Gyro-Bohm diffusivity [m^2/s], the unit QLKNN's fluxes are expressed in.

    ``chi_GB = sqrt(m_i) T_i^1.5 / ((e B)^2 a)``, as in TORAX's
    ``calculate_chiGB`` with the ion temperature and minor radius as
    references. It is the natural size of turbulent diffusion for given
    temperature, field and machine size.
    """
    return ((A_i * M_AMU) ** 0.5 / (B0 * Q_E) ** 2
            * (Ti_keV * KEV_TO_J) ** 1.5 / a_minor)


def compute_inputs(geom: FluxSurfaceGeometry, k: KineticAssumptions,
                   torax_corrections: bool = True) -> dict[str, np.ndarray]:
    """QLKNN_7_11's ten inputs on every flux surface of ``geom``.

    Returns a dict with the ten inputs under :data:`INPUT_NAMES`, plus the
    raw ``x_unscaled``, the MHD ``alpha``, and the kinetic profiles, for
    reporting.

    With ``torax_corrections`` (TORAX's defaults), three adjustments TORAX
    makes before calling QLKNN are applied: shear is reduced by ``alpha/2``
    (the main effect of the plasma's own pressure on the field), a ``q < 1``
    core is treated as ``q = 1, smag = 0.1`` (a stand-in for sawtooth
    crashes), and ``smag - alpha`` is kept above -0.2.
    """
    r = geom.r_mid
    Te, Ti, ne = kinetic_profiles(geom.rho, k)
    # Ion dilution for a single carbon impurity (charge 6), a common default:
    # quasineutrality and the Z_eff definition give n_i/n_e = (6 - Z_eff)/5.
    normni = np.full_like(ne, (6.0 - k.Z_eff) / 5.0)
    ni = normni * ne

    def norm_grad(f: np.ndarray) -> np.ndarray:
        # -R d(ln f)/dr, TORAX's lref_over_l with R_major as the reference.
        return -geom.R_major * np.gradient(np.log(f), r)

    Ati, Ate, Ane = norm_grad(Ti), norm_grad(Te), norm_grad(ne)
    Ani = Ane.copy()
    q = geom.q.copy()
    smag = r / q * np.gradient(q, r)

    alpha = (2 * KEV_TO_J * MU_0 / geom.B0**2 * q**2
             * (Te * ne * (Ate + Ane) + Ti * ni * (Ati + Ani)))
    if torax_corrections:
        smag = smag - alpha / 2
        sawtooth = q < 1
        smag = np.where(sawtooth, 0.1, smag)
        q = np.where(sawtooth, 1.0, q)
        smag = np.where(smag - alpha < -0.2, alpha - 0.2, smag)

    x_unscaled = np.maximum(r / geom.a_minor, 1e-9)
    eps_lcfs = geom.a_minor / geom.R_major
    x = x_unscaled * eps_lcfs / EPSILON_NN

    return {
        "Ati": Ati, "Ate": Ate, "Ane": Ane, "Ani": Ani, "q": q, "smag": smag,
        "x": x, "Ti_Te": Ti / Te,
        "LogNuStar": log_nu_star(Te, ne, geom.q, geom.R_mid, geom.epsilon, k.Z_eff),
        "normni": normni,
        "x_unscaled": x_unscaled, "alpha": alpha, "Te_keV": Te, "Ti_keV": Ti,
        "ne": ne,
    }


def input_matrix(inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Stack the ten inputs into the (n_surfaces, 10) array QLKNN takes."""
    return np.stack([np.asarray(inputs[n], dtype=float) for n in INPUT_NAMES], axis=1)


def training_range_report(inputs: dict[str, np.ndarray]) -> dict:
    """Which surfaces sit inside a box QLKNN was trained on, and which inputs fail.

    A surface counts as in-distribution if *all ten* inputs fall inside the
    same training box. Returns per-surface booleans, the fraction inside, and
    for the core box, how many surfaces each input pushes outside.
    """
    n = len(inputs["q"])
    inside_any = np.zeros(n, dtype=bool)
    failures: dict[str, dict[str, int]] = {}
    for box_name, box in TRAINING_BOXES.items():
        ok = np.ones(n, dtype=bool)
        fails = {}
        for name, (lo, hi) in box.items():
            v = np.asarray(inputs[name])
            tol = 1e-6 * max(1.0, abs(hi))
            bad = (v < lo - tol) | (v > hi + tol)
            fails[name] = int(bad.sum())
            ok &= ~bad
        failures[box_name] = fails
        inside_any |= ok
    return {
        "inside": inside_any,
        "fraction_inside": float(inside_any.mean()),
        "failures_by_box": failures,
    }


def combine_fluxes(fluxes: dict[str, np.ndarray],
                   etg_correction: float = 1.0 / 3.0) -> dict[str, np.ndarray]:
    """Sum QLKNN's per-mode fluxes the way TORAX does (gyro-Bohm units).

    QLKNN predicts heat flux separately for three kinds of turbulence:
    ITG (driven by the ion temperature gradient), TEM (trapped electrons) and
    ETG (electron temperature gradient, at much smaller scales). TORAX scales
    ETG by 1/3 by default before adding them up.
    """
    def f(name: str) -> np.ndarray:
        return np.asarray(fluxes[name], dtype=float).reshape(-1)

    return {
        "qi": f("efiITG") + f("efiTEM"),
        "qe": f("efeITG") + f("efeTEM") + etg_correction * f("efeETG"),
        "pfe": f("pfeITG") + f("pfeTEM"),
        "gamma_max": f("gamma_max"),
    }


def heat_diffusivity(q_gb: np.ndarray, grad: np.ndarray, chi_gb: np.ndarray,
                     R_major: float, a_minor: float) -> np.ndarray:
    """Convert a gyro-Bohm heat flux into a diffusivity chi [m^2/s].

    Follows TORAX's ``_make_core_transport``:
    ``chi = (R/a) * Q_GB / (R/L_T) * chi_GB``. Where the gradient is near zero
    chi is undefined and NaN is returned.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = (R_major / a_minor) * q_gb / grad * chi_gb
    return np.where(np.abs(grad) > 1e-6, chi, np.nan)
