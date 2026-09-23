"""Train/test split and data-leakage tests for the surrogate.

Every model comparison in this repo relies on these properties: one frozen
test set per (n_samples, test_frac, seed), and nothing about the test rows
reaching the trained weights or the normalization statistics.
"""
import numpy as np

from src.ml.dataset import generate_dataset
from src.ml.mlp import MLPRegressor
from src.ml.surrogate import split_indices, train_surrogate


def test_split_matches_recorded_indices():
    """Pinned to the indices train_surrogate produced before the split was
    factored out, so the refactor cannot silently change which samples are
    held out."""
    train, test = split_indices(50, 0.2, 0)
    assert test.tolist() == [15, 25, 37, 23, 30, 6, 27, 49, 3, 28]
    assert len(train) == 40


def test_split_is_disjoint_and_complete():
    for n, frac, seed in [(50, 0.2, 0), (1001, 0.25, 3), (8000, 0.2, 0)]:
        train, test = split_indices(n, frac, seed)
        assert len(np.intersect1d(train, test)) == 0
        assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(n))
        assert len(test) == int(round(frac * n))


def test_split_is_deterministic_and_seed_dependent():
    a = split_indices(200, 0.2, 5)
    b = split_indices(200, 0.2, 5)
    c = split_indices(200, 0.2, 6)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


N_SAMPLES, EPOCHS, SEED = 300, 3, 4


def test_normalization_stats_come_from_training_rows_only():
    model, _, metrics = train_surrogate(n_samples=N_SAMPLES, epochs=EPOCHS, seed=SEED)
    X, y, _ = generate_dataset(n_samples=N_SAMPLES, seed=SEED)
    train, test = split_indices(N_SAMPLES, 0.2, SEED)

    assert metrics.n_train == len(train) and metrics.n_test == len(test)
    np.testing.assert_array_equal(model.x_mean_, X[train].mean(0))
    np.testing.assert_array_equal(model.y_mean_, y[train].mean(0))
    # Guard against the check passing trivially: the full-data mean differs.
    assert not np.allclose(model.x_mean_, X.mean(0), rtol=0, atol=0)


def test_test_rows_do_not_change_trained_weights():
    """train_surrogate passes test rows to fit() for loss history only.
    A model fit on the training rows alone must be identical."""
    model, _, _ = train_surrogate(n_samples=N_SAMPLES, epochs=EPOCHS, seed=SEED)
    X, y, _ = generate_dataset(n_samples=N_SAMPLES, seed=SEED)
    train, _ = split_indices(N_SAMPLES, 0.2, SEED)
    alone = MLPRegressor(epochs=EPOCHS, seed=SEED).fit(X[train], y[train])

    for Wa, Wb in zip(model.weights, alone.weights):
        np.testing.assert_array_equal(Wa, Wb)
