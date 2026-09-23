"""Tests for the PyTorch surrogate, checked against the NumPy reference."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.checks import sensor_ids  # noqa: E402
from src.ml.dataset import PARAM_NAMES, generate_dataset  # noqa: E402
from src.ml.mlp import MLPRegressor, r2_score  # noqa: E402
from src.ml.torch_mlp import TorchMLPRegressor, mask_inputs  # noqa: E402
from src.ml.uncertainty import EnsembleSurrogate, mc_dropout_predict  # noqa: E402
from src.ml.validation import apply_dropout  # noqa: E402


@pytest.fixture(scope="module")
def data():
    X, y, _ = generate_dataset(n_samples=300, seed=0)
    return X, y


@pytest.fixture(scope="module")
def fitted(data):
    X, y = data
    return TorchMLPRegressor(hidden_layers=(32, 32), epochs=5, seed=0).fit(X, y)


def _linear_problem(n=400):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 5))
    return X, X @ rng.standard_normal((5, 2)) + 0.1


# --- construction and shapes ----------------------------------------------------

def test_rejects_bad_settings():
    with pytest.raises(ValueError, match="dtype"):
        TorchMLPRegressor(dtype="float16")
    with pytest.raises(ValueError, match="input_dropout"):
        TorchMLPRegressor(input_dropout=1.0)


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA present")
def test_missing_cuda_fails_loudly():
    with pytest.raises(RuntimeError, match="CUDA"):
        TorchMLPRegressor(device="cuda")


def test_runs_on_cpu_by_default(fitted):
    assert all(p.device.type == "cpu" for p in fitted.net.parameters())
    assert all(p.dtype == torch.float64 for p in fitted.net.parameters())


def test_predict_shapes_and_dtype(fitted, data):
    X, _ = data
    out = fitted.predict(X[:7])
    assert out.shape == (7, len(PARAM_NAMES)) and out.dtype == np.float64
    assert fitted.predict(X[:1]).shape == (1, len(PARAM_NAMES))


def test_architecture_matches_request(fitted, data):
    X, _ = data
    widths = [(lyr.in_features, lyr.out_features) for lyr in fitted.net.layers]
    assert widths == [(X.shape[1], 32), (32, 32), (32, len(PARAM_NAMES))]


# --- equivalence with the NumPy reference ---------------------------------------

def test_same_weights_give_same_predictions(data):
    X, y = data
    ref = MLPRegressor(hidden_layers=(32, 32), epochs=3, seed=1).fit(X, y)
    tm = TorchMLPRegressor.from_numpy(ref)
    np.testing.assert_allclose(tm.predict(X), ref.predict(X), rtol=1e-12, atol=1e-9)


def test_training_steps_match_numpy(data):
    """From identical weights, five full-batch Adam steps give identical weights.

    Full batch removes batch order (the two frameworks shuffle with different
    RNGs). l2 is large on purpose: Adam is invariant to a constant gradient
    scale, so only a sizeable L2 term exposes a wrongly scaled loss, decay on
    the biases, or AdamW's decoupled decay.
    """
    X, y = data
    kw = dict(hidden_layers=(16, 16), seed=2, l2=0.1, batch_size=len(X))
    start = MLPRegressor(epochs=0, **kw).fit(X, y)
    after = MLPRegressor(epochs=5, **kw).fit(X, y)
    tm = TorchMLPRegressor.from_numpy(start)
    tm._run_epochs(X, y, 5, torch.Generator().manual_seed(0))
    for layer, W, b in zip(tm.net.layers, after.weights, after.biases):
        np.testing.assert_allclose(layer.weight.detach().numpy().T, W, atol=1e-12)
        np.testing.assert_allclose(layer.bias.detach().numpy(), b, atol=1e-12)


def test_learns_linear_map_like_numpy():
    X, y = _linear_problem()
    model = TorchMLPRegressor(hidden_layers=(32,), epochs=200, seed=0).fit(X, y, X, y)
    assert r2_score(y, model.predict(X)) > 0.98
    assert model.history_[-1] < model.history_[0] / 10


# --- training behavior ----------------------------------------------------------

def test_same_seed_is_deterministic(data):
    X, y = data
    a = TorchMLPRegressor(hidden_layers=(16,), epochs=3, seed=5, input_dropout=0.2).fit(X, y)
    b = TorchMLPRegressor(hidden_layers=(16,), epochs=3, seed=5, input_dropout=0.2).fit(X, y)
    c = TorchMLPRegressor(hidden_layers=(16,), epochs=3, seed=6, input_dropout=0.2).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert not np.allclose(a.predict(X), c.predict(X))


def test_he_initialization(data):
    """Weights start N(0, 2/fan_in) and biases at zero, as in mlp.py.

    The parity tests copy weights from NumPy, so they never exercise the
    PyTorch initializer; this checks it directly.
    """
    X, y = data
    model = TorchMLPRegressor(hidden_layers=(256, 256), epochs=0, seed=0).fit(X, y)
    for layer in model.net.linears[:-1]:
        w = layer.weight.detach().numpy()
        expected = np.sqrt(2.0 / layer.in_features)
        assert abs(w.std() / expected - 1) < 0.05
        assert abs(w.mean()) < 0.05 * expected
        assert not layer.bias.detach().numpy().any()


def test_one_epoch_changes_weights(data):
    X, y = data
    start = MLPRegressor(hidden_layers=(16,), epochs=0, seed=0).fit(X, y)
    tm = TorchMLPRegressor.from_numpy(start)
    before = [p.detach().clone() for p in tm.net.parameters()]
    tm._run_epochs(X, y, 1, torch.Generator().manual_seed(0))
    assert all(not torch.equal(b, p) for b, p in zip(before, tm.net.parameters()))


def test_normalization_from_training_rows_and_val_rows_ignored(data):
    X, y = data
    tr, va = slice(0, 200), slice(200, 300)
    with_val = TorchMLPRegressor(hidden_layers=(16,), epochs=3, seed=0).fit(
        X[tr], y[tr], X[va], y[va])
    without = TorchMLPRegressor(hidden_layers=(16,), epochs=3, seed=0).fit(X[tr], y[tr])
    np.testing.assert_array_equal(with_val.x_mean_, X[tr].mean(0))
    np.testing.assert_array_equal(with_val.predict(X), without.predict(X))
    assert len(with_val.history_) == 3 and without.history_ == []


# --- sensor dropout -------------------------------------------------------------

def test_mask_inputs_zeroes_without_rescaling():
    x = torch.full((2000, 50), 3.0, dtype=torch.float64)
    out = mask_inputs(x, 0.2, torch.Generator().manual_seed(0))
    values = set(out.unique().tolist())
    assert values == {0.0, 3.0}, "kept entries must not be rescaled by 1/(1-p)"
    assert abs(float((out == 0).double().mean()) - 0.2) < 0.01


def test_works_with_existing_dropout_and_uncertainty_code(data):
    """The robustness and MC-dropout code only needs x_mean_ and predict."""
    X, y = data
    model = TorchMLPRegressor(hidden_layers=(16,), epochs=3, seed=0,
                              input_dropout=0.15).fit(X, y)
    Xd = apply_dropout(X[:10], model, 0.2, np.random.default_rng(0))
    assert model.predict(Xd).shape == (10, len(PARAM_NAMES))
    mu, sd = mc_dropout_predict(model, X[:10], n_passes=10)
    assert mu.shape == sd.shape == (10, len(PARAM_NAMES)) and np.mean(sd) > 0


def test_ensemble_of_torch_members(data):
    X, y = data
    members = [TorchMLPRegressor(hidden_layers=(16,), epochs=3, seed=k).fit(X, y)
               for k in range(3)]
    mu, sd = EnsembleSurrogate(models=members).predict_with_std(X[:5])
    assert mu.shape == sd.shape == (5, len(PARAM_NAMES)) and np.all(sd > 0)


# --- input validation -----------------------------------------------------------

def test_predict_rejects_wrong_feature_count(fitted, data):
    X, _ = data
    with pytest.raises(ValueError, match=f"expected {X.shape[1]}"):
        fitted.predict(X[:, :-1])


def test_predict_rejects_one_dimensional_input(fitted, data):
    X, _ = data
    with pytest.raises(ValueError, match="2-D"):
        fitted.predict(X[0])


def test_rejects_nonfinite(fitted, data):
    X, y = data
    Xbad = X[:4].copy()
    Xbad[2, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fitted.predict(Xbad)
    ybad = y.copy()
    ybad[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        TorchMLPRegressor(epochs=1).fit(X, ybad)


def test_rejects_mismatched_sample_counts(data):
    X, y = data
    with pytest.raises(ValueError, match="samples"):
        TorchMLPRegressor(epochs=1).fit(X, y[:-1])


# --- schema and persistence -----------------------------------------------------

def test_names_follow_project_schema(fitted):
    assert fitted.feature_names_ == sensor_ids()
    assert fitted.target_names_ == list(PARAM_NAMES)


def test_save_load_roundtrip(fitted, data, tmp_path):
    X, _ = data
    path = tmp_path / "m.pt"
    fitted.save(path)
    reloaded = TorchMLPRegressor.load(path)
    np.testing.assert_array_equal(reloaded.predict(X), fitted.predict(X))
    assert reloaded.hidden_layers == fitted.hidden_layers
    assert reloaded.input_dropout == fitted.input_dropout


def test_load_rejects_different_sensor_order(data, tmp_path):
    X, y = data
    ids = sensor_ids()
    ids[0], ids[1] = ids[1], ids[0]
    model = TorchMLPRegressor(hidden_layers=(8,), epochs=1).fit(X, y, feature_names=ids)
    model.save(tmp_path / "m.pt")
    with pytest.raises(ValueError, match="sensor order differs at position 0"):
        TorchMLPRegressor.load(tmp_path / "m.pt")


def test_load_rejects_foreign_or_future_files(fitted, tmp_path):
    torch.save({"format": "something-else"}, tmp_path / "a.pt")
    with pytest.raises(ValueError, match="not a"):
        TorchMLPRegressor.load(tmp_path / "a.pt")
    fitted.save(tmp_path / "b.pt")
    ckpt = torch.load(tmp_path / "b.pt", weights_only=True)
    ckpt["version"] = 99
    torch.save(ckpt, tmp_path / "b.pt")
    with pytest.raises(ValueError, match="version 99"):
        TorchMLPRegressor.load(tmp_path / "b.pt")


def test_float32_mode_close_to_float64(data):
    X, y = data
    ref = MLPRegressor(hidden_layers=(16,), epochs=2, seed=0).fit(X, y)
    t32 = TorchMLPRegressor.from_numpy(ref, dtype="float32")
    np.testing.assert_allclose(t32.predict(X), ref.predict(X), rtol=1e-4)
