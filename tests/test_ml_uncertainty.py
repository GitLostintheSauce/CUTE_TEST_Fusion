"""Tests for surrogate uncertainty quantification and calibration."""
import numpy as np
import pytest

from src.ml.dataset import PARAM_NAMES, generate_dataset
from src.ml.surrogate import train_surrogate
from src.ml.uncertainty import (
    NOMINAL_1SIGMA,
    EnsembleSurrogate,
    calibration_report,
    fit_variance_scaling,
    mc_dropout_predict,
    train_ensemble,
)


def _small_ensemble():
    return train_ensemble(n_members=3, n_samples=1200, epochs=100, seed=0)


def test_ensemble_predict_shapes_and_positive_std():
    ens, layout = _small_ensemble()
    X, y, _ = generate_dataset(n_samples=40, seed=7, layout=layout)
    mu, sd = ens.predict_with_std(X)

    assert mu.shape == y.shape
    assert sd.shape == y.shape
    assert np.all(sd >= 0)
    # Members must actually disagree, otherwise the error bars are vacuous.
    assert np.mean(sd) > 0


def test_ensemble_mean_matches_predict():
    ens, layout = _small_ensemble()
    X, _, _ = generate_dataset(n_samples=20, seed=8, layout=layout)
    assert np.allclose(ens.predict(X), ens.predict_with_std(X)[0])


def test_ensemble_save_load_roundtrip(tmp_path):
    ens, layout = _small_ensemble()
    X, _, _ = generate_dataset(n_samples=15, seed=9, layout=layout)
    ens.save(tmp_path / "ens")
    reloaded = EnsembleSurrogate.load(tmp_path / "ens")

    assert len(reloaded.models) == len(ens.models)
    mu_a, sd_a = ens.predict_with_std(X)
    mu_b, sd_b = reloaded.predict_with_std(X)
    assert np.allclose(mu_a, mu_b)
    assert np.allclose(sd_a, sd_b)


def test_ensemble_load_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        EnsembleSurrogate.load(tmp_path / "nope")


def test_calibration_report_bounds():
    ens, layout = _small_ensemble()
    X, y, _ = generate_dataset(n_samples=120, seed=11, layout=layout)
    mu, sd = ens.predict_with_std(X)
    cal = calibration_report(y, mu, sd)

    for name in PARAM_NAMES:
        assert 0.0 <= cal.coverage_1sigma[name] <= 1.0
        # Two sigma must cover at least as much as one sigma.
        assert cal.coverage_2sigma[name] >= cal.coverage_1sigma[name]
        assert cal.mean_sigma[name] > 0
        assert cal.rmse[name] >= 0


def test_variance_scaling_moves_coverage_toward_nominal():
    """The headline claim: recalibration makes the error bars honest."""
    ens, layout = _small_ensemble()
    X_cal, y_cal, _ = generate_dataset(n_samples=400, seed=21, layout=layout)
    X_test, y_test, _ = generate_dataset(n_samples=400, seed=22, layout=layout)

    mu_c, sd_c = ens.predict_with_std(X_cal)
    mu_t, sd_t = ens.predict_with_std(X_test)
    scales = fit_variance_scaling(y_cal, mu_c, sd_c)

    before = calibration_report(y_test, mu_t, sd_t)
    after = calibration_report(y_test, mu_t, sd_t * scales)

    assert scales.shape == (len(PARAM_NAMES),)
    assert np.all(scales > 0)

    # Averaged over parameters, calibrated coverage sits closer to nominal.
    err_before = np.mean([abs(before.coverage_1sigma[n] - NOMINAL_1SIGMA)
                          for n in PARAM_NAMES])
    err_after = np.mean([abs(after.coverage_1sigma[n] - NOMINAL_1SIGMA)
                         for n in PARAM_NAMES])
    assert err_after < err_before


def test_mc_dropout_requires_nonzero_rate():
    model, layout, _ = train_surrogate(n_samples=800, epochs=60, seed=0)
    X, _, _ = generate_dataset(n_samples=10, seed=31, layout=layout)
    with pytest.raises(ValueError):
        mc_dropout_predict(model, X)


def test_mc_dropout_produces_spread():
    model, layout, _ = train_surrogate(n_samples=1200, epochs=100, seed=0,
                                       input_dropout=0.15)
    X, y, _ = generate_dataset(n_samples=30, seed=32, layout=layout)
    mu, sd = mc_dropout_predict(model, X, n_passes=20)

    assert mu.shape == y.shape
    assert np.all(sd >= 0)
    assert np.mean(sd) > 0
