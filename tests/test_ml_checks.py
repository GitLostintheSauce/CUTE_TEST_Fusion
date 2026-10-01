"""Tests for surrogate boundary checks and the sensor feature schema."""
import numpy as np
import pytest

from src.forward.sensors import generate_cute_sensors
from src.ml.checks import (
    check_features,
    check_norm_stats,
    check_schema,
    check_targets,
    sensor_ids,
)
from src.ml.dataset import PARAM_NAMES, SensorLayout, generate_dataset
from src.ml.mlp import MLPRegressor
from src.ml.surrogate import predict_parameters

N = SensorLayout.from_config().n_sensors


# --- schema -------------------------------------------------------------------

def test_sensor_ids_match_dataset_width():
    X, _, _ = generate_dataset(n_samples=3, seed=0)
    ids = sensor_ids()
    assert len(ids) == X.shape[1] == N
    assert len(set(ids)) == len(ids), "sensor ids must be unique"


def test_sensor_id_order_matches_layout_coordinates():
    """Feature i must be the sensor whose coordinates the layout uses at i."""
    config = generate_cute_sensors()
    layout = SensorLayout.from_config(config)
    ids = sensor_ids(config)
    n_fl = len(layout.fl_R)
    assert all(i.startswith(("FS", "FC")) for i in ids[:n_fl])
    assert all(i.startswith("S") for i in ids[n_fl:])
    for k, s in enumerate(config.flux_loops):
        assert (layout.fl_R[k], layout.fl_Z[k]) == (s["R"], s["Z"])
    for k, s in enumerate(config.mirnov_probes):
        assert (layout.mp_R[k], layout.mp_Z[k]) == (s["R"], s["Z"])


def test_check_schema_accepts_current_order():
    check_schema(sensor_ids(), list(PARAM_NAMES))


def test_check_schema_rejects_swapped_sensors():
    ids = sensor_ids()
    ids[3], ids[4] = ids[4], ids[3]
    with pytest.raises(ValueError, match="sensor order differs at position 3"):
        check_schema(ids, list(PARAM_NAMES))


def test_check_schema_rejects_reordered_targets():
    with pytest.raises(ValueError, match="target order"):
        check_schema(sensor_ids(), ["R0", "Ip", "Z0", "a"])


def test_check_schema_rejects_wrong_sensor_count():
    with pytest.raises(ValueError, match=f"{N - 1} sensors"):
        check_schema(sensor_ids()[:-1], list(PARAM_NAMES))


# --- features and targets -----------------------------------------------------

def test_check_features_returns_float64():
    out = check_features(np.ones((2, N), dtype=np.int32), N)
    assert out.dtype == np.float64 and out.shape == (2, N)


@pytest.mark.parametrize("shape", [(N,), (1, N - 1), (1, N + 1), (2, 3, N)])
def test_check_features_rejects_bad_shapes(shape):
    with pytest.raises(ValueError):
        check_features(np.zeros(shape), N)


def test_check_features_names_the_count():
    with pytest.raises(ValueError, match=f"{N - 1} sensor signals per sample; expected {N}"):
        check_features(np.zeros((1, N - 1)), N)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_check_features_rejects_nonfinite(bad):
    X = np.zeros((3, N))
    X[1, 7] = bad
    with pytest.raises(ValueError, match="sample 1, column 7"):
        check_features(X, N)


@pytest.mark.parametrize("arr", [np.zeros((1, N), dtype=bool),
                                 np.zeros((1, N), dtype=complex),
                                 np.array([["a"] * N])])
def test_check_features_rejects_non_real_dtypes(arr):
    with pytest.raises(ValueError, match="real-valued"):
        check_features(arr, N)


def test_check_features_rejects_empty():
    with pytest.raises(ValueError, match="no samples"):
        check_features(np.zeros((0, N)), N)


def test_check_targets_uses_param_count():
    check_targets(np.zeros((2, len(PARAM_NAMES))))
    with pytest.raises(ValueError):
        check_targets(np.zeros((2, len(PARAM_NAMES) + 1)))


def test_check_norm_stats():
    check_norm_stats(np.zeros(4), np.ones(4), 4, "y")
    with pytest.raises(ValueError, match="shape"):
        check_norm_stats(np.zeros(3), np.ones(4), 4, "y")
    with pytest.raises(ValueError, match="positive"):
        check_norm_stats(np.zeros(4), np.array([1.0, 0.0, 1.0, 1.0]), 4, "y")
    with pytest.raises(ValueError, match="non-finite"):
        check_norm_stats(np.array([0.0, np.nan, 0, 0]), np.ones(4), 4, "y")


# --- predict_parameters boundary ---------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    X, y, _ = generate_dataset(n_samples=60, seed=0)
    return MLPRegressor(hidden_layers=(8,), epochs=2, seed=0).fit(X, y)


def test_predict_parameters_rejects_wrong_length(tiny_model):
    with pytest.raises(ValueError, match=f"expected {N}"):
        predict_parameters(tiny_model, np.zeros(N - 1))


def test_predict_parameters_rejects_nan(tiny_model):
    signals = np.zeros(N)
    signals[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        predict_parameters(tiny_model, signals)
