"""Tests for the QLKNN coupling proof of concept (src/coupling/).

The input calculation is checked against analytic profiles and an independent
textbook collision time, so the numbers fed to QLKNN are known to be what the
formulas say. The model itself is only exercised when the optional
``fusion_surrogates`` extra is installed.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.coupling import qlknn_runner
from src.coupling.qlknn_inputs import (
    EPSILON_NN,
    INPUT_NAMES,
    KEV_TO_J,
    M_E,
    FluxSurfaceGeometry,
    KineticAssumptions,
    combine_fluxes,
    compute_inputs,
    heat_diffusivity,
    input_matrix,
    kinetic_profiles,
    log_nu_star,
    training_range_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def circular_geometry(a_over_R: float = 1 / 3, n: int = 400) -> FluxSurfaceGeometry:
    """Concentric circular surfaces with q = 1 + 2 rho^2."""
    R0, B0 = 0.3, 0.5
    a = a_over_R * R0
    rho = np.linspace(0.02, 0.98, n)
    r = rho * a
    return FluxSurfaceGeometry(psi_norm=rho**2, q=1 + 2 * rho**2, R_in=R0 - r,
                               R_out=R0 + r, R_major=R0, a_minor=a, B0=B0)


def test_temperature_gradient_matches_analytic():
    geom = circular_geometry()
    k = KineticAssumptions()
    inp = compute_inputs(geom, k, torax_corrections=False)
    rho = geom.rho
    Te, _, _ = kinetic_profiles(rho, k)
    dT_drho = ((k.Te0_keV - k.Te_edge_keV) * k.T_peaking
               * (1 - rho**2) ** (k.T_peaking - 1) * (-2 * rho))
    expected = -geom.R_major * dT_drho / geom.a_minor / Te
    interior = slice(5, -5)
    np.testing.assert_allclose(inp["Ate"][interior], expected[interior], rtol=2e-3)
    # Ti is a constant multiple of Te, so its log-gradient is identical.
    np.testing.assert_allclose(inp["Ati"], inp["Ate"])


def test_magnetic_shear_matches_analytic():
    geom = circular_geometry()
    inp = compute_inputs(geom, KineticAssumptions(), torax_corrections=False)
    rho = geom.rho
    expected = 4 * rho**2 / (1 + 2 * rho**2)   # (r/q) dq/dr for q = 1 + 2 rho^2
    np.testing.assert_allclose(inp["smag"][5:-5], expected[5:-5], atol=1e-4)


@pytest.mark.parametrize("a_over_R", [1 / 3, 0.6])
def test_x_is_rescaled_by_aspect_ratio(a_over_R):
    geom = circular_geometry(a_over_R)
    inp = compute_inputs(geom, KineticAssumptions())
    np.testing.assert_allclose(inp["x"], geom.rho * a_over_R / EPSILON_NN)
    np.testing.assert_allclose(inp["x_unscaled"], geom.rho)


def test_collisionality_matches_wesson_collision_time():
    # Independent form: tau_e = 1.09e16 Te[keV]^1.5 / (n_e lnLambda) seconds
    # (Wesson, Tokamaks, 3rd ed.), then nu* = q R / (eps^1.5 v_te tau_e).
    Te, ne, q, R, eps, Z = 0.1, 1e19, 2.0, 0.32, 0.2, 1.0
    lnL = 31.3 - 0.5 * np.log(ne) + np.log(Te * 1e3)
    tau_e = 1.09e16 * Te**1.5 / (ne * lnL)
    v_te = np.sqrt(Te * KEV_TO_J / M_E)
    expected = np.log10(Z / tau_e * q * R / (eps**1.5 * v_te))
    got = log_nu_star(np.array([Te]), np.array([ne]), np.array([q]),
                      np.array([R]), np.array([eps]), Z)[0]
    assert got == pytest.approx(expected, abs=0.01)


def test_torax_corrections_apply_sawtooth_proxy():
    geom = circular_geometry()
    geom = FluxSurfaceGeometry(**{**geom.__dict__, "q": geom.q - 0.5})  # q < 1 in core
    inp = compute_inputs(geom, KineticAssumptions())
    core = geom.q < 1
    assert core.any()
    np.testing.assert_allclose(inp["q"][core], 1.0)
    np.testing.assert_allclose(inp["smag"][core], 0.1)


def _inputs_at(**values):
    base = {"Ati": 5.0, "Ate": 5.0, "Ane": 1.0, "Ani": 1.0, "q": 2.0, "smag": 1.0,
            "x": 0.5, "Ti_Te": 1.0, "LogNuStar": -2.0, "normni": 1.0}
    base.update(values)
    return {k: np.array([v]) for k, v in base.items()}


def test_range_report_core_box():
    assert training_range_report(_inputs_at())["inside"][0]
    rep = training_range_report(_inputs_at(x=1.2))
    assert not rep["inside"][0]
    assert rep["failures_by_box"]["QLKNN11D (core)"]["x"] == 1


def test_range_report_uses_boxes_separately():
    # Inside the edge dataset's box (x fixed at 0.95, Ti = Te, steep gradients).
    edge = _inputs_at(x=0.95, Ati=50.0, Ate=50.0, Ane=20.0, Ani=20.0, q=5.0,
                      smag=10.0, LogNuStar=0.0)
    assert training_range_report(edge)["inside"][0]
    # Inside the union of both boxes but in neither one: what the model's own
    # inputs_and_ranges would wrongly accept.
    gap = _inputs_at(x=0.5, Ati=100.0)
    assert not training_range_report(gap)["inside"][0]


def test_input_matrix_order():
    inp = _inputs_at()
    X = input_matrix(inp)
    assert X.shape == (1, len(INPUT_NAMES))
    assert X[0, INPUT_NAMES.index("x")] == 0.5


def test_combine_fluxes_scales_etg():
    one = np.array([1.0])
    fl = {n: one for n in ("efiITG", "efiTEM", "efeITG", "efeTEM", "efeETG",
                           "pfeITG", "pfeTEM", "gamma_max")}
    tot = combine_fluxes(fl)
    assert tot["qi"][0] == 2.0
    assert tot["qe"][0] == pytest.approx(2.0 + 1 / 3)


def test_heat_diffusivity_nan_without_gradient():
    chi = heat_diffusivity(np.array([1.0, 1.0]), np.array([0.0, 2.0]),
                           np.array([3.0, 3.0]), R_major=0.3, a_minor=0.15)
    assert np.isnan(chi[0])
    assert chi[1] == pytest.approx(2.0 * 1.0 / 2.0 * 3.0)


def test_geometry_round_trip():
    geom = circular_geometry(n=10)
    back = FluxSurfaceGeometry.from_dict(json.loads(json.dumps(geom.to_dict())))
    np.testing.assert_allclose(back.q, geom.q)
    assert back.a_minor == geom.a_minor


def test_committed_equilibrium_is_fatter_than_qlknn_training():
    """The proof of concept's central geometric finding, checked on its data."""
    eq = json.loads((PROJECT_ROOT / "models" / "qlknn_poc_equilibrium.json").read_text())
    geom = FluxSurfaceGeometry.from_dict(eq["geometry"])
    a_over_R = geom.a_minor / geom.R_major
    assert a_over_R > 0.5
    rho_limit = 0.95 * EPSILON_NN / a_over_R
    assert 0.4 < rho_limit < 0.6
    assert np.all(np.diff(geom.rho) > 0)


def test_predict_fluxes_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        qlknn_runner.predict_fluxes(object(), np.zeros((3, 9)))


def test_qlknn_model_runs_on_cute_inputs():
    pytest.importorskip("fusion_surrogates")
    model = qlknn_runner.load_model()
    eq = json.loads((PROJECT_ROOT / "models" / "qlknn_poc_equilibrium.json").read_text())
    inp = compute_inputs(FluxSurfaceGeometry.from_dict(eq["geometry"]), KineticAssumptions())
    fluxes = qlknn_runner.predict_fluxes(model, input_matrix(inp))
    assert {"efiITG", "efeETG", "gamma_max"} <= set(fluxes)
    for v in fluxes.values():
        assert v.shape == (len(inp["q"]),)
        assert np.all(np.isfinite(v))
