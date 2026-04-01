"""Validation benchmarks for the reconstruction pipeline."""
from __future__ import annotations

import numpy as np


def snr_to_sigma(snr_db: float, signal_rms: float) -> float:
    """Convert SNR in dB to noise standard deviation."""
    if snr_db == float("inf"):
        return 0.0
    return signal_rms / (10 ** (snr_db / 20))


def noise_sweep(
    clean_measurements: dict[str, float],
    snr_levels: list[float],
    seed: int = 42,
) -> dict[float, dict[str, float]]:
    """Apply noise at multiple SNR levels to clean measurements.

    Args:
        clean_measurements: Dict of sensor_id -> clean value.
        snr_levels: List of SNR values in dB (use float('inf') for no noise).
        seed: Random seed for reproducibility.

    Returns:
        Dict of snr_db -> noisy measurement dict.
    """
    signal_rms = np.sqrt(np.mean([v**2 for v in clean_measurements.values()]))

    results = {}
    for snr_db in snr_levels:
        sigma = snr_to_sigma(snr_db, signal_rms)
        rng = np.random.default_rng(seed)

        noisy = {}
        for sensor_id, value in clean_measurements.items():
            if sigma > 0:
                noisy[sensor_id] = value + rng.normal(0, sigma)
            else:
                noisy[sensor_id] = value
        results[snr_db] = noisy

    return results


def sensor_dropout(
    measurements: dict[str, float],
    fraction: float,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Remove a fraction of sensors from measurements.

    Args:
        measurements: Dict of sensor_id -> value.
        fraction: Fraction of sensors to remove (0.0 to 1.0).
        rng: Random number generator.

    Returns:
        Dict with some sensors removed.
    """
    if rng is None:
        rng = np.random.default_rng()

    sensor_ids = list(measurements.keys())
    n_remove = int(len(sensor_ids) * fraction)
    remove_ids = set(rng.choice(sensor_ids, size=n_remove, replace=False))

    return {k: v for k, v in measurements.items() if k not in remove_ids}
