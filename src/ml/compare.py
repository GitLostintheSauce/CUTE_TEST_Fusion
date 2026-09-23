"""Backend-agnostic scoring and timing for comparing surrogate implementations.

Works with any model that has ``predict(X) -> ndarray`` and ``x_mean_``, so
the NumPy and PyTorch surrogates are scored by the same code on the same
data. A comparison where each side computes its own metrics is not a fair
comparison.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.ml.dataset import PARAM_NAMES
from src.ml.mlp import r2_score
from src.ml.validation import apply_dropout


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    """R2, MAE and RMSE per parameter (physical units), plus overall R2."""
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(PARAM_NAMES):
        err = y_pred[:, j] - y_true[:, j]
        out[name] = {
            "r2": r2_score(y_true[:, j], y_pred[:, j]),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
        }
    out["overall"] = {"r2": r2_score(y_true, y_pred)}
    return out


def time_predict(model: Any, X: np.ndarray, repeats: int = 20, warmup: int = 3) -> float:
    """Median wall time of ``model.predict(X)`` in seconds per sample.

    Procedure: ``warmup`` untimed calls (first-call allocation and dispatch
    costs), then ``repeats`` timed calls on the same batch, reporting the
    median so one scheduler hiccup cannot move the number. Compare two
    models only when both are timed this way, on the same X, on the same
    machine and thread count.
    """
    for _ in range(warmup):
        model.predict(X)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) / len(X)


def dropout_r2(model: Any, X: np.ndarray, y: np.ndarray, frac: float,
               seeds: range | list[int] = range(3)) -> float:
    """Overall R2 with a fraction of sensors dead, averaged over dropout draws.

    Dead sensors are mean-imputed per sample by
    :func:`src.ml.validation.apply_dropout`, the same definition used by the
    validation report and notebook 03.
    """
    scores = []
    for s in seeds:
        Xd = apply_dropout(X, model, frac, np.random.default_rng(s))
        scores.append(r2_score(y, model.predict(Xd)))
    return float(np.mean(scores))
