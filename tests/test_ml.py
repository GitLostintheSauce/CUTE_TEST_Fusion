"""Tests for the ML equilibrium-reconstruction surrogate."""
import numpy as np
import pytest

from src.ml.baseline import invert_least_squares
from src.ml.dataset import (
    PARAM_NAMES,
    SensorLayout,
    forward_signals,
    generate_dataset,
)
from src.ml.mlp import MLPRegressor, r2_score
from src.ml.physics import MU0, biot_savart_field, loop_field, loop_flux
from src.ml.surrogate import predict_parameters, train_surrogate

# --- physics ----------------------------------------------------------------

def test_on_axis_field_matches_exact():
    """B_z on the loop axis matches the exact analytic formula."""
    a, curr = 0.3, 1.0e5
    for Z in [0.0, 0.1, 0.25]:
        _, bz = loop_field(a, 0.0, curr, 1e-9, Z)
        exact = MU0 * curr * a ** 2 / (2.0 * (a ** 2 + Z ** 2) ** 1.5)
        assert bz == pytest.approx(exact, rel=1e-6)


def test_analytic_field_matches_biot_savart():
    """Analytic loop field agrees with direct Biot-Savart quadrature off-axis."""
    a, curr = 0.3, 1.0e5
    for R, Z in [(0.45, 0.05), (0.2, 0.15), (0.5, -0.2)]:
        br, bz = loop_field(a, 0.0, curr, R, Z)
        bbr, bbz = biot_savart_field(a, 0.0, curr, R, Z)
        assert br == pytest.approx(bbr, rel=2e-3, abs=1e-6)
        assert bz == pytest.approx(bbz, rel=2e-3, abs=1e-6)


def test_flux_decays_with_distance():
    """Poloidal flux from a loop decreases far from it."""
    near = loop_flux(0.3, 0.0, 1e5, 0.45, 0.0)
    far = loop_flux(0.3, 0.0, 1e5, 1.5, 0.0)
    assert abs(near) > abs(far)


# --- dataset ----------------------------------------------------------------

def test_dataset_shapes_and_determinism():
    X1, y1, layout = generate_dataset(n_samples=50, seed=3)
    assert X1.shape == (50, 130)
    assert y1.shape == (50, 4)
    X2, y2, _ = generate_dataset(n_samples=50, seed=3)
    assert np.allclose(X1, X2) and np.allclose(y1, y2)


def test_forward_signal_length():
    layout = SensorLayout.from_config()
    vec = forward_signals(1.5e5, 0.32, 0.0, 0.12, layout)
    assert vec.shape == (layout.n_sensors,)
    assert np.all(np.isfinite(vec))


# --- MLP --------------------------------------------------------------------

def test_mlp_learns_linear_map():
    """The MLP recovers a known linear mapping with high R2."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 5))
    W = rng.standard_normal((5, 2))
    y = X @ W + 0.1
    model = MLPRegressor(hidden_layers=(32,), epochs=200, seed=0).fit(X, y)
    assert r2_score(y, model.predict(X)) > 0.98


def test_mlp_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 4))
    y = X @ rng.standard_normal((4, 1))
    model = MLPRegressor(hidden_layers=(16,), epochs=50, seed=0).fit(X, y)
    path = tmp_path / "m.npz"
    model.save(str(path))
    reloaded = MLPRegressor.load(str(path))
    assert np.allclose(model.predict(X), reloaded.predict(X))


# --- surrogate + baseline ---------------------------------------------------

def test_surrogate_trains_to_good_accuracy():
    """End-to-end surrogate reaches strong held-out R2 on all parameters."""
    _, _, metrics = train_surrogate(n_samples=2000, epochs=150, seed=0)
    assert metrics.r2_overall > 0.9
    for name in PARAM_NAMES:
        assert metrics.r2_per_param[name] > 0.8, f"{name} R2 too low"


def test_baseline_recovers_parameters():
    """The iterative baseline recovers known parameters from clean signals."""
    layout = SensorLayout.from_config()
    true = np.array([1.5e5, 0.32, 0.01, 0.12])
    signals = forward_signals(*true, layout)
    est = invert_least_squares(signals, layout)
    assert est[0] == pytest.approx(true[0], rel=0.02)   # Ip within 2%
    assert est[1] == pytest.approx(true[1], abs=0.005)  # R0 within 5 mm
    assert est[3] == pytest.approx(true[3], abs=0.01)   # a within 10 mm


def test_predict_parameters_returns_named_dict():
    model, layout, _ = train_surrogate(n_samples=1500, epochs=120, seed=0)
    signals = forward_signals(1.4e5, 0.31, 0.0, 0.11, layout)
    pred = predict_parameters(model, signals)
    assert set(pred.keys()) == set(PARAM_NAMES)
    assert 2.0e4 < pred["Ip"] < 2.5e5
