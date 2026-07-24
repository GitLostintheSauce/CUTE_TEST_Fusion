"""Tests for the ML surrogate robustness validation studies."""
import numpy as np

from src.ml.dataset import generate_dataset
from src.ml.surrogate import train_surrogate
from src.ml.validation import (
    apply_dropout,
    dropout_sweep,
    noise_sweep,
    run_validation,
)


def _small_model(input_dropout=0.0):
    return train_surrogate(n_samples=1200, epochs=120, seed=0,
                           input_dropout=input_dropout)


def test_apply_dropout_masks_expected_channel_count():
    """Dropout replaces the requested fraction of channels with the mean."""
    model, layout, _ = _small_model()
    X, _, _ = generate_dataset(n_samples=5, seed=1, layout=layout)
    rng = np.random.default_rng(0)
    X_drop = apply_dropout(X, model, 0.2, rng)

    assert X_drop.shape == X.shape
    n_changed = [np.sum(~np.isclose(X[i], X_drop[i])) for i in range(len(X))]
    expected = int(round(0.2 * X.shape[1]))
    # Each row loses about the requested number of channels (a masked channel
    # could coincidentally already equal the mean, so allow a small margin).
    for n in n_changed:
        assert abs(n - expected) <= 2


def test_apply_dropout_zero_frac_is_identity():
    model, layout, _ = _small_model()
    X, _, _ = generate_dataset(n_samples=3, seed=2, layout=layout)
    out = apply_dropout(X, model, 0.0, np.random.default_rng(0))
    assert np.allclose(out, X)


def test_noise_sweep_degrades_monotonically_overall():
    """Accuracy should not improve as noise grows far beyond training noise."""
    model, layout, _ = _small_model()
    points = noise_sweep(model, layout, noise_levels=(0.0, 0.05, 0.20),
                         n_eval=150)
    assert len(points) == 3
    assert points[0].r2_overall > points[-1].r2_overall


def test_dropout_sweep_returns_requested_points():
    model, layout, _ = _small_model()
    fracs = (0.0, 0.1, 0.3)
    points = dropout_sweep(model, layout, dropout_fracs=fracs, n_eval=150)
    assert [p.setting for p in points] == list(fracs)
    assert points[0].r2_overall > points[-1].r2_overall


def test_dropout_augmentation_improves_dropout_robustness():
    """The headline claim: training with masked inputs survives dead sensors."""
    base, layout, _ = _small_model(input_dropout=0.0)
    robust, _, _ = _small_model(input_dropout=0.15)

    frac = (0.2,)
    base_r2 = dropout_sweep(base, layout, dropout_fracs=frac, n_eval=250)[0]
    robust_r2 = dropout_sweep(robust, layout, dropout_fracs=frac,
                              n_eval=250)[0]
    assert robust_r2.r2_overall > base_r2.r2_overall


def test_run_validation_report_serializes():
    model, layout, _ = _small_model()
    report = run_validation(model, layout, n_eval=100)
    d = report.as_dict()
    assert "noise_sweep" in d and "dropout_sweep" in d
    assert len(d["noise_sweep"]) > 0
    assert "r2_overall" in d["noise_sweep"][0]
