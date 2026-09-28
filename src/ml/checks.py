"""Boundary checks for surrogate inputs, targets, and saved models.

NumPy broadcasting will happily subtract a 64-element mean from a
63-element vector in some shapes, or carry a NaN from one dead probe into
every predicted parameter. These checks turn those cases into errors that
say what was expected and what arrived.

The expected sensor count and ordering come from the sensor configuration
(:func:`sensor_ids`), not from a hard-coded number.
"""
from __future__ import annotations

import numpy as np

from src.forward.sensors import SensorConfig, generate_cute_sensors
from src.ml.dataset import PARAM_NAMES


def sensor_ids(config: SensorConfig | None = None) -> list[str]:
    """Sensor identifiers in feature order: flux loops, then Mirnov probes.

    This is the order :func:`src.ml.dataset.forward_signals` concatenates
    signals in, so feature ``i`` of every X array is the sensor ``ids[i]``.
    """
    config = config or generate_cute_sensors()
    return ([s["id"] for s in config.flux_loops]
            + [s["id"] for s in config.mirnov_probes])


def _check_matrix(arr_like, n_cols: int, name: str, what: str) -> np.ndarray:
    arr = np.asarray(arr_like)
    if arr.dtype == bool or not np.issubdtype(arr.dtype, np.number) \
            or np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f"{name} must be real-valued numbers; got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D with shape (n_samples, {n_cols}); got shape "
            f"{arr.shape}. Wrap a single sample as X[None, :]."
        )
    if arr.shape[0] == 0:
        raise ValueError(f"{name} has no samples")
    if arr.shape[1] != n_cols:
        raise ValueError(
            f"{name} has {arr.shape[1]} {what} per sample; expected {n_cols}"
        )
    arr = arr.astype(np.float64, copy=False)
    bad = ~np.isfinite(arr)
    if bad.any():
        rows, cols = np.nonzero(bad)
        raise ValueError(
            f"{name} contains {int(bad.sum())} non-finite values (first at "
            f"sample {rows[0]}, column {cols[0]})"
        )
    return arr


def check_features(X, n_features: int, name: str = "X") -> np.ndarray:
    """Validate a sensor matrix and return it as float64.

    Dead sensors are represented in this project by mean imputation (see
    :func:`src.ml.validation.apply_dropout`), never by NaN, so any NaN here
    is a data error rather than a missing probe.
    """
    return _check_matrix(X, n_features, name, "sensor signals")


def check_targets(y, n_targets: int = len(PARAM_NAMES), name: str = "y") -> np.ndarray:
    """Validate a target matrix ordered as :data:`PARAM_NAMES`."""
    return _check_matrix(y, n_targets, name, "target values")


def check_norm_stats(mean, std, n: int, name: str) -> None:
    """Validate a (mean, std) pair used for standardization."""
    for label, arr in (("mean", mean), ("std", std)):
        arr = np.asarray(arr)
        if arr.shape != (n,):
            raise ValueError(
                f"{name} {label} has shape {arr.shape}; expected ({n},)"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} {label} contains non-finite values")
    if np.any(np.asarray(std) <= 0):
        raise ValueError(f"{name} std must be strictly positive")


def check_schema(
    saved_sensor_ids: list[str],
    saved_param_names: list[str],
    expected_sensor_ids: list[str] | None = None,
    expected_param_names: list[str] | None = None,
) -> None:
    """Check a saved model was trained on the same feature and target order.

    A model trained on a different sensor layout can have the right input
    width and still be wrong, because feature ``i`` would mean a different
    probe. Comparing identifiers catches that; comparing counts does not.
    """
    expected_sensor_ids = expected_sensor_ids or sensor_ids()
    expected_param_names = expected_param_names or list(PARAM_NAMES)
    for label, saved, expected in (
        ("sensor", list(saved_sensor_ids), list(expected_sensor_ids)),
        ("target", list(saved_param_names), list(expected_param_names)),
    ):
        if len(saved) != len(expected):
            raise ValueError(
                f"saved model has {len(saved)} {label}s; current schema has "
                f"{len(expected)}"
            )
        for i, (s, e) in enumerate(zip(saved, expected)):
            if s != e:
                raise ValueError(
                    f"{label} order differs at position {i}: saved model has "
                    f"{s!r}, current schema has {e!r}"
                )
