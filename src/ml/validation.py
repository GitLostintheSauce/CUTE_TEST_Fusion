"""Robustness validation for the ML equilibrium-reconstruction surrogate.

Two studies a reviewer will actually ask about:

1. **Noise robustness.** How does reconstruction accuracy degrade as the
   diagnostic measurement noise grows? Real magnetics are noisy, so a
   surrogate that only works on clean signals is not useful.

2. **Sensor dropout.** Magnetic probes fail. If a fraction of the 130
   channels goes dead, does the reconstruction fall apart or degrade
   gracefully?

Both studies report numbers, not adjectives. Dropout is simulated at
inference time by replacing dead channels with the model's own training-set
channel means (standard mean-imputation, which represents "this channel
carries no information"). The surrogate is *not* retrained with dropout
augmentation, so these figures measure graceful degradation of an
unmodified model, which is the honest and harder test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.ml.dataset import PARAM_NAMES, SensorLayout, generate_dataset
from src.ml.mlp import MLPRegressor, r2_score


@dataclass
class SweepPoint:
    """Accuracy at one setting of a swept variable."""

    setting: float
    r2_overall: float
    r2_per_param: dict[str, float]
    mae_per_param: dict[str, float]


@dataclass
class ValidationReport:
    """Full robustness report for a trained surrogate."""

    noise_sweep: list[SweepPoint] = field(default_factory=list)
    dropout_sweep: list[SweepPoint] = field(default_factory=list)
    train_noise_frac: float = 0.02
    n_eval: int = 0

    def as_dict(self) -> dict:
        def pack(points):
            return [
                {
                    "setting": p.setting,
                    "r2_overall": p.r2_overall,
                    "r2_per_param": p.r2_per_param,
                    "mae_per_param": p.mae_per_param,
                }
                for p in points
            ]

        return {
            "train_noise_frac": self.train_noise_frac,
            "n_eval": self.n_eval,
            "noise_sweep": pack(self.noise_sweep),
            "dropout_sweep": pack(self.dropout_sweep),
        }


def _score(y_true: np.ndarray, y_pred: np.ndarray, setting: float) -> SweepPoint:
    r2_per, mae_per = {}, {}
    for j, name in enumerate(PARAM_NAMES):
        r2_per[name] = r2_score(y_true[:, j], y_pred[:, j])
        mae_per[name] = float(np.mean(np.abs(y_true[:, j] - y_pred[:, j])))
    return SweepPoint(
        setting=setting,
        r2_overall=r2_score(y_true, y_pred),
        r2_per_param=r2_per,
        mae_per_param=mae_per,
    )


def noise_sweep(
    model: MLPRegressor,
    layout: SensorLayout,
    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20),
    n_eval: int = 400,
    seed: int = 4242,
) -> list[SweepPoint]:
    """Evaluate accuracy across increasing sensor-noise fractions."""
    points = []
    for k, noise in enumerate(noise_levels):
        X, y, _ = generate_dataset(n_samples=n_eval, noise_frac=noise,
                                   seed=seed + k, layout=layout)
        points.append(_score(y, model.predict(X), noise))
    return points


def apply_dropout(
    X: np.ndarray,
    model: MLPRegressor,
    frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate dead channels by mean-imputing a random fraction of sensors.

    Each sample gets its own random set of dead channels, mimicking
    independent probe failures across shots.
    """
    if frac <= 0:
        return X
    assert model.x_mean_ is not None, "model must be fit"
    X_out = X.copy()
    n_sensors = X.shape[1]
    n_dead = int(round(frac * n_sensors))
    if n_dead == 0:
        return X_out
    for i in range(X.shape[0]):
        dead = rng.choice(n_sensors, size=n_dead, replace=False)
        X_out[i, dead] = model.x_mean_[dead]
    return X_out


def dropout_sweep(
    model: MLPRegressor,
    layout: SensorLayout,
    dropout_fracs: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50),
    n_eval: int = 400,
    noise_frac: float = 0.02,
    seed: int = 909,
) -> list[SweepPoint]:
    """Evaluate accuracy as a growing fraction of sensors goes dead."""
    X, y, _ = generate_dataset(n_samples=n_eval, noise_frac=noise_frac,
                               seed=seed, layout=layout)
    points = []
    for k, frac in enumerate(dropout_fracs):
        rng = np.random.default_rng(seed + 100 + k)
        X_drop = apply_dropout(X, model, frac, rng)
        points.append(_score(y, model.predict(X_drop), frac))
    return points


def run_validation(
    model: MLPRegressor,
    layout: SensorLayout,
    n_eval: int = 400,
    train_noise_frac: float = 0.02,
) -> ValidationReport:
    """Run both robustness studies and return a report."""
    return ValidationReport(
        noise_sweep=noise_sweep(model, layout, n_eval=n_eval),
        dropout_sweep=dropout_sweep(model, layout, n_eval=n_eval,
                                    noise_frac=train_noise_frac),
        train_noise_frac=train_noise_frac,
        n_eval=n_eval,
    )
