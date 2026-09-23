"""High-level train / predict API for the equilibrium-reconstruction surrogate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ml.checks import check_features
from src.ml.dataset import PARAM_NAMES, SensorLayout, generate_dataset
from src.ml.mlp import MLPRegressor, r2_score


@dataclass
class SurrogateMetrics:
    """Per-parameter and aggregate accuracy on a held-out test set."""

    r2_overall: float
    r2_per_param: dict[str, float]
    mae_per_param: dict[str, float]
    n_train: int
    n_test: int

    def as_dict(self) -> dict:
        return {
            "r2_overall": self.r2_overall,
            "r2_per_param": self.r2_per_param,
            "mae_per_param": self.mae_per_param,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


def split_indices(
    n: int, test_frac: float = 0.2, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) for a dataset of ``n`` samples.

    Seeded with ``seed + 1`` so the split is independent of the data draw,
    which :func:`generate_dataset` seeds with ``seed``. Every model compared
    on one dataset must take its split from here so they share a test set.
    """
    n_test = int(round(test_frac * n))
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n)
    return perm[n_test:], perm[:n_test]


def score_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, n_train: int,
) -> SurrogateMetrics:
    """Held-out R2 and MAE, per parameter and overall, in physical units."""
    r2_per = {}
    mae_per = {}
    for j, name in enumerate(PARAM_NAMES):
        r2_per[name] = r2_score(y_true[:, j], y_pred[:, j])
        mae_per[name] = float(np.mean(np.abs(y_true[:, j] - y_pred[:, j])))
    return SurrogateMetrics(
        r2_overall=r2_score(y_true, y_pred),
        r2_per_param=r2_per,
        mae_per_param=mae_per,
        n_train=n_train,
        n_test=len(y_true),
    )


def train_surrogate(
    n_samples: int = 4000,
    noise_frac: float = 0.02,
    test_frac: float = 0.2,
    seed: int = 0,
    epochs: int = 300,
    hidden_layers: tuple[int, ...] = (128, 128),
    input_dropout: float = 0.0,
) -> tuple[MLPRegressor, SensorLayout, SurrogateMetrics]:
    """Generate data, train the MLP surrogate, and evaluate on a held-out split.

    Args:
        input_dropout: Sensor-failure augmentation; see
            :class:`src.ml.mlp.MLPRegressor`. Set above 0 to train a model
            that degrades gracefully when diagnostic channels go dead.

    Returns:
        (trained model, sensor layout, held-out metrics).
    """
    X, y, layout = generate_dataset(n_samples=n_samples, noise_frac=noise_frac,
                                    seed=seed)
    train_idx, test_idx = split_indices(len(X), test_frac, seed)

    model = MLPRegressor(hidden_layers=hidden_layers, epochs=epochs, seed=seed,
                         input_dropout=input_dropout)
    # The test rows only feed the per-epoch loss history; fit() never uses
    # them to update weights or to stop early.
    model.fit(X[train_idx], y[train_idx], X[test_idx], y[test_idx])

    metrics = score_predictions(y[test_idx], model.predict(X[test_idx]),
                                n_train=len(train_idx))
    return model, layout, metrics


def predict_parameters(model: MLPRegressor, signals: np.ndarray) -> dict[str, float]:
    """Run the surrogate on a single sensor vector, returning named parameters."""
    if model.x_mean_ is None:
        raise RuntimeError("Model must be fit or loaded before prediction.")
    signals = check_features(np.asarray(signals).reshape(1, -1),
                             len(model.x_mean_), name="signals")
    pred = model.predict(signals)[0]
    return {name: float(pred[j]) for j, name in enumerate(PARAM_NAMES)}
