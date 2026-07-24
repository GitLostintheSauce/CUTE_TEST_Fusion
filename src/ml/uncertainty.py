"""Uncertainty quantification for the equilibrium-reconstruction surrogate.

A point prediction of "Ip = 184 kA" is much less useful to an operator than
"Ip = 184 kA plus or minus 3 kA". This module produces error bars two ways
and, importantly, **checks whether those error bars are honest**.

Two estimators:

* **Deep ensemble.** Train K networks from different random seeds on
  different bootstrap-style shuffles. The spread of their predictions
  estimates uncertainty. This is the standard strong baseline for
  regression UQ (Lakshminarayanan et al., 2017).
* **MC dropout.** Run a single dropout-trained network many times with
  input masking active, and take the spread across stochastic passes.
  Much cheaper than an ensemble since it needs only one trained model.

Calibration is not optional. An uncertainty estimate that does not track
the actual error is decoration, so :func:`calibration_report` measures
empirical coverage (what fraction of true values land inside the nominal
1-sigma and 2-sigma bands) and the correlation between predicted sigma and
realized absolute error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from src.ml.dataset import PARAM_NAMES, SensorLayout, generate_dataset
from src.ml.mlp import MLPRegressor


@dataclass
class EnsembleSurrogate:
    """A deep ensemble of surrogates giving mean predictions and error bars."""

    models: list[MLPRegressor]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble mean prediction."""
        return self.predict_with_std(X)[0]

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, standard deviation) across ensemble members."""
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return preds.mean(axis=0), preds.std(axis=0, ddof=1)

    def save(self, directory) -> None:
        from pathlib import Path
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.models):
            m.save(str(directory / f"member_{i}.npz"))

    @classmethod
    def load(cls, directory) -> "EnsembleSurrogate":
        from pathlib import Path
        directory = Path(directory)
        paths = sorted(directory.glob("member_*.npz"))
        if not paths:
            raise FileNotFoundError(f"No ensemble members found in {directory}")
        return cls(models=[MLPRegressor.load(str(p)) for p in paths])


def train_ensemble(
    n_members: int = 5,
    n_samples: int = 6000,
    noise_frac: float = 0.02,
    epochs: int = 300,
    seed: int = 0,
    hidden_layers: tuple[int, ...] = (128, 128),
) -> tuple[EnsembleSurrogate, SensorLayout]:
    """Train an ensemble, each member on its own data draw and seed.

    Varying both the data draw and the initialization is what makes the
    members disagree in a useful way; identical training would collapse the
    spread to zero and produce meaningless error bars.
    """
    layout = SensorLayout.from_config()
    models = []
    for k in range(n_members):
        X, y, _ = generate_dataset(n_samples=n_samples, noise_frac=noise_frac,
                                   seed=seed + 1000 * (k + 1), layout=layout)
        model = MLPRegressor(hidden_layers=hidden_layers, epochs=epochs,
                             seed=seed + k)
        model.fit(X, y)
        models.append(model)
    return EnsembleSurrogate(models), layout


def mc_dropout_predict(
    model: MLPRegressor,
    X: np.ndarray,
    n_passes: int = 50,
    dropout: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate (mean, std) by repeated stochastic forward passes.

    Input channels are randomly masked on each pass, mimicking the sensor
    dropout the robust model was trained with. Requires a model trained with
    ``input_dropout`` above zero for the spread to be meaningful.

    Args:
        model: Trained surrogate (ideally dropout-augmented).
        X: Sensor vectors, shape (n_samples, n_sensors).
        n_passes: Number of stochastic passes.
        dropout: Masking rate; defaults to the model's training rate.
        seed: RNG seed.
    """
    rate = model.input_dropout if dropout is None else dropout
    if rate <= 0:
        raise ValueError(
            "MC dropout needs a nonzero masking rate; train the model with "
            "input_dropout > 0 or pass dropout explicitly."
        )
    assert model.x_mean_ is not None, "model must be fit"
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)

    preds = []
    for _ in range(n_passes):
        keep = rng.random(X.shape) >= rate
        # Masked channels fall back to the training mean, matching how the
        # model saw them during training.
        X_masked = np.where(keep, X, model.x_mean_)
        preds.append(model.predict(X_masked))
    stacked = np.stack(preds, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=1)


@dataclass
class CalibrationResult:
    """Per-parameter calibration diagnostics."""

    coverage_1sigma: dict[str, float]
    coverage_2sigma: dict[str, float]
    mean_sigma: dict[str, float]
    rmse: dict[str, float]
    sigma_error_corr: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "coverage_1sigma": self.coverage_1sigma,
            "coverage_2sigma": self.coverage_2sigma,
            "mean_sigma": self.mean_sigma,
            "rmse": self.rmse,
            "sigma_error_corr": self.sigma_error_corr,
        }


# Nominal coverage of a Gaussian within 1 and 2 standard deviations.
NOMINAL_1SIGMA = float(2 * norm.cdf(1.0) - 1.0)   # ~0.6827
NOMINAL_2SIGMA = float(2 * norm.cdf(2.0) - 1.0)   # ~0.9545


def fit_variance_scaling(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    level: float = NOMINAL_1SIGMA,
) -> np.ndarray:
    """Fit one scale factor per parameter so error bars match nominal coverage.

    Raw ensemble spread is rarely calibrated: it tends to be too wide for some
    parameters and too narrow for others. For each parameter we take the
    empirical distribution of the ratio ``|error| / sigma`` and pick the
    quantile at the desired coverage level. Multiplying sigma by that factor
    makes empirical coverage match nominal by construction.

    This is a nonparametric variance recalibration. It **must** be fit on a
    calibration split that is separate from the data used to report the final
    calibration numbers, otherwise the result is circular.

    Returns:
        Array of per-parameter scale factors, shape (n_params,).
    """
    scales = np.empty(len(PARAM_NAMES))
    for j in range(len(PARAM_NAMES)):
        err = np.abs(y_true[:, j] - y_pred[:, j])
        sig = np.maximum(y_std[:, j], 1e-12)
        scales[j] = float(np.quantile(err / sig, level))
    return scales


def calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> CalibrationResult:
    """Measure whether the predicted error bars are trustworthy.

    Returns empirical coverage at 1 and 2 sigma (compare against
    :data:`NOMINAL_1SIGMA` and :data:`NOMINAL_2SIGMA`), the average predicted
    sigma versus the realized RMSE, and the correlation between predicted
    sigma and absolute error. A useful uncertainty estimate has coverage near
    nominal and a clearly positive correlation.
    """
    cov1, cov2, msig, rmse, corr = {}, {}, {}, {}, {}
    for j, name in enumerate(PARAM_NAMES):
        err = np.abs(y_true[:, j] - y_pred[:, j])
        sig = np.maximum(y_std[:, j], 1e-12)
        cov1[name] = float(np.mean(err <= sig))
        cov2[name] = float(np.mean(err <= 2.0 * sig))
        msig[name] = float(np.mean(sig))
        rmse[name] = float(np.sqrt(np.mean(
            (y_true[:, j] - y_pred[:, j]) ** 2)))
        if np.std(sig) < 1e-15:
            corr[name] = float("nan")
        else:
            corr[name] = float(np.corrcoef(sig, err)[0, 1])
    return CalibrationResult(cov1, cov2, msig, rmse, corr)
