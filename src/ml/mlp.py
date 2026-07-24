"""A small multilayer perceptron implemented from scratch in NumPy.

No deep-learning framework is required: forward pass, backprop, the Adam
optimizer, and input/output standardization are all implemented here. This
keeps the surrogate dependency-light and fully reproducible, and it makes the
learning machinery inspectable rather than hidden behind a library call.

The network is a standard regression MLP: fully connected layers with ReLU
activations and a linear output head, trained to minimize mean squared error
on standardized targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MLPRegressor:
    """Fully connected regression network with ReLU hidden layers and Adam.

    Args:
        hidden_layers: Sizes of the hidden layers.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        epochs: Training epochs.
        seed: RNG seed for weight init and shuffling.
        l2: L2 weight-decay coefficient.
        input_dropout: Fraction of input channels randomly zeroed (in
            standardized space, so zero equals the channel mean) on each
            training batch. This is augmentation for sensor failure: it
            teaches the network to reconstruct from partial diagnostics.
    """

    hidden_layers: tuple[int, ...] = (128, 128)
    lr: float = 3e-3
    batch_size: int = 128
    epochs: int = 300
    seed: int = 0
    l2: float = 1e-6
    input_dropout: float = 0.0

    weights: list[np.ndarray] = field(default_factory=list, repr=False)
    biases: list[np.ndarray] = field(default_factory=list, repr=False)
    x_mean_: np.ndarray | None = field(default=None, repr=False)
    x_std_: np.ndarray | None = field(default=None, repr=False)
    y_mean_: np.ndarray | None = field(default=None, repr=False)
    y_std_: np.ndarray | None = field(default=None, repr=False)
    history_: list[float] = field(default_factory=list, repr=False)

    # -- internals -----------------------------------------------------------

    def _init_params(self, n_in: int, n_out: int, rng: np.random.Generator):
        sizes = [n_in, *self.hidden_layers, n_out]
        self.weights, self.biases = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            # He initialization for ReLU layers.
            self.weights.append(rng.standard_normal((a, b)) * np.sqrt(2.0 / a))
            self.biases.append(np.zeros(b))

    def _forward(self, X):
        """Return (output, list of pre-activations, list of activations)."""
        activations = [X]
        pre = []
        h = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            pre.append(z)
            if i < len(self.weights) - 1:
                h = np.maximum(z, 0.0)  # ReLU
            else:
                h = z                    # linear output
            activations.append(h)
        return h, pre, activations

    # -- public API ----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "MLPRegressor":
        """Train on (X, y). Standardizes inputs and targets internally."""
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]

        self.x_mean_, self.x_std_ = X.mean(0), X.std(0) + 1e-12
        self.y_mean_, self.y_std_ = y.mean(0), y.std(0) + 1e-12
        Xs = (X - self.x_mean_) / self.x_std_
        ys = (y - self.y_mean_) / self.y_std_

        self._init_params(Xs.shape[1], ys.shape[1], rng)

        # Adam moment buffers.
        mW = [np.zeros_like(W) for W in self.weights]
        vW = [np.zeros_like(W) for W in self.weights]
        mb = [np.zeros_like(b) for b in self.biases]
        vb = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        n = Xs.shape[0]

        for _ in range(self.epochs):
            perm = rng.permutation(n)
            Xs_sh, ys_sh = Xs[perm], ys[perm]
            for start in range(0, n, self.batch_size):
                xb = Xs_sh[start:start + self.batch_size]
                yb = ys_sh[start:start + self.batch_size]
                t += 1

                if self.input_dropout > 0.0:
                    # Zero in standardized space == impute the channel mean,
                    # matching how dead sensors are handled at inference.
                    keep = rng.random(xb.shape) >= self.input_dropout
                    xb = xb * keep

                out, pre, acts = self._forward(xb)
                m = xb.shape[0]
                grad = (2.0 / m) * (out - yb)  # dMSE/dout

                gW: list = [None] * len(self.weights)
                gb: list = [None] * len(self.biases)
                for i in reversed(range(len(self.weights))):
                    gW[i] = acts[i].T @ grad + self.l2 * self.weights[i]
                    gb[i] = grad.sum(0)
                    if i > 0:
                        grad = grad @ self.weights[i].T
                        grad = grad * (pre[i - 1] > 0.0)  # ReLU derivative

                for i in range(len(self.weights)):
                    mW[i] = beta1 * mW[i] + (1 - beta1) * gW[i]
                    vW[i] = beta2 * vW[i] + (1 - beta2) * (gW[i] ** 2)
                    mb[i] = beta1 * mb[i] + (1 - beta1) * gb[i]
                    vb[i] = beta2 * vb[i] + (1 - beta2) * (gb[i] ** 2)
                    mW_hat = mW[i] / (1 - beta1 ** t)
                    vW_hat = vW[i] / (1 - beta2 ** t)
                    mb_hat = mb[i] / (1 - beta1 ** t)
                    vb_hat = vb[i] / (1 - beta2 ** t)
                    self.weights[i] -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)
                    self.biases[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

            if X_val is not None and y_val is not None:
                self.history_.append(float(
                    np.mean((self.predict(X_val) - np.asarray(
                        y_val, dtype=float).reshape(len(y_val), -1)) ** 2)
                ))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets in original (unstandardized) units."""
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_std_
        out, _, _ = self._forward(Xs)
        return out * self.y_std_ + self.y_mean_

    # -- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        """Save weights, biases, and normalization stats to a .npz file."""
        if (self.x_mean_ is None or self.x_std_ is None
                or self.y_mean_ is None or self.y_std_ is None):
            raise RuntimeError("Model must be fit before saving.")
        arrs: dict[str, np.ndarray] = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            arrs[f"W{i}"] = W
            arrs[f"b{i}"] = b
        arrs["x_mean"] = self.x_mean_
        arrs["x_std"] = self.x_std_
        arrs["y_mean"] = self.y_mean_
        arrs["y_std"] = self.y_std_
        arrs["n_layers"] = np.array(len(self.weights))
        np.savez(path, **arrs)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str) -> "MLPRegressor":
        """Load a model saved with :meth:`save`."""
        data = np.load(path, allow_pickle=False)
        model = cls()
        n = int(data["n_layers"])
        model.weights = [data[f"W{i}"] for i in range(n)]
        model.biases = [data[f"b{i}"] for i in range(n)]
        model.x_mean_, model.x_std_ = data["x_mean"], data["x_std"]
        model.y_mean_, model.y_std_ = data["y_mean"], data["y_std"]
        return model


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2, averaged over targets."""
    y_true = np.asarray(y_true, dtype=float).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(len(y_pred), -1)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(0)) ** 2, axis=0) + 1e-12
    return float(np.mean(1.0 - ss_res / ss_tot))
