"""Noise models for synthetic sensor data."""
from __future__ import annotations

import numpy as np


def add_white_noise(
    values: np.ndarray, sigma: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Add Gaussian white noise."""
    if rng is None:
        rng = np.random.default_rng()
    return values + rng.normal(0, sigma, values.shape)


def add_60hz_pickup(
    values: np.ndarray, amplitude: float, timestamps: np.ndarray
) -> np.ndarray:
    """Add 60 Hz sinusoidal pickup."""
    return values + amplitude * np.sin(2 * np.pi * 60.0 * timestamps)


def add_dropout(
    values: np.ndarray, probability: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Replace random samples with NaN (channel dropout)."""
    if rng is None:
        rng = np.random.default_rng()
    result = values.copy()
    mask = rng.random(values.shape) < probability
    result[mask] = np.nan
    return result


def apply_noise_model(
    values: np.ndarray,
    timestamps: np.ndarray,
    white_sigma: float = 0.0,
    pickup_amplitude: float = 0.0,
    dropout_probability: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply full noise model to a signal."""
    result = values.copy()
    if white_sigma > 0:
        result = add_white_noise(result, white_sigma, rng)
    if pickup_amplitude > 0:
        result = add_60hz_pickup(result, pickup_amplitude, timestamps)
    if dropout_probability > 0:
        result = add_dropout(result, dropout_probability, rng)
    return result
