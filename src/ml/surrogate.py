"""High-level train / predict API for the equilibrium-reconstruction surrogate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    n_test = int(round(test_frac * len(X)))
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(len(X))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    model = MLPRegressor(hidden_layers=hidden_layers, epochs=epochs, seed=seed,
                         input_dropout=input_dropout)
    model.fit(X[train_idx], y[train_idx], X[test_idx], y[test_idx])

    y_pred = model.predict(X[test_idx])
    y_true = y[test_idx]
    r2_per = {}
    mae_per = {}
    for j, name in enumerate(PARAM_NAMES):
        r2_per[name] = r2_score(y_true[:, j], y_pred[:, j])
        mae_per[name] = float(np.mean(np.abs(y_true[:, j] - y_pred[:, j])))

    metrics = SurrogateMetrics(
        r2_overall=r2_score(y_true, y_pred),
        r2_per_param=r2_per,
        mae_per_param=mae_per,
        n_train=len(train_idx),
        n_test=len(test_idx),
    )
    return model, layout, metrics


def predict_parameters(model: MLPRegressor, signals: np.ndarray) -> dict[str, float]:
    """Run the surrogate on a single sensor vector, returning named parameters."""
    signals = np.asarray(signals, dtype=float).reshape(1, -1)
    pred = model.predict(signals)[0]
    return {name: float(pred[j]) for j, name in enumerate(PARAM_NAMES)}
