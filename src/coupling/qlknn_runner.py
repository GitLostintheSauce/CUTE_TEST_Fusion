"""Load and run QLKNN_7_11 from Google DeepMind's ``fusion_surrogates``.

Kept separate from :mod:`src.coupling.qlknn_inputs` so that computing and
checking the inputs needs only NumPy. ``fusion_surrogates`` (and JAX, which it
runs on) is imported only when a prediction is actually requested, and is an
optional extra: ``pip install -e ".[qlknn]"``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.coupling.qlknn_inputs import INPUT_NAMES

PINNED_VERSION = "0.4.7"


def load_model() -> Any:
    """Load the default QLKNN_7_11 model, checking its inputs match ours.

    Raises:
        ImportError: ``fusion_surrogates`` is not installed.
        RuntimeError: the installed model expects different inputs, or in a
            different order, than :data:`INPUT_NAMES`. Feeding columns in the
            wrong order would run silently and return nonsense.
    """
    import jax

    # Match the precision of fusion_surrogates' own numerical tests.
    jax.config.update("jax_enable_x64", True)
    from fusion_surrogates.qlknn import qlknn_model

    model = qlknn_model.QLKNNModel.load_default_model()
    if list(model.config.input_names) != INPUT_NAMES:
        raise RuntimeError(
            f"QLKNN expects inputs {model.config.input_names}, this module "
            f"provides {INPUT_NAMES}. fusion_surrogates {PINNED_VERSION} is the "
            "version this was written against."
        )
    return model


def predict_fluxes(model: Any, X: np.ndarray) -> dict[str, np.ndarray]:
    """Run the model on an (n, 10) input array; return fluxes in gyro-Bohm units."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(INPUT_NAMES):
        raise ValueError(f"X must have shape (n, {len(INPUT_NAMES)}); got {X.shape}")
    out = model.predict(X)
    return {name: np.asarray(v, dtype=float).reshape(-1) for name, v in out.items()}
