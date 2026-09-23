"""PyTorch version of the surrogate MLP in :mod:`src.ml.mlp`.

Same network, same training recipe, same public interface as
:class:`src.ml.mlp.MLPRegressor`: NumPy arrays go in and come out, and the
fitted model exposes ``x_mean_``, ``input_dropout`` and ``predict``, so the
existing robustness and uncertainty code runs on it unchanged. The NumPy
model stays in the repo as the reference this one is checked against.

What PyTorch replaces: the hand-written backward pass (autograd) and the
hand-written Adam update (``torch.optim.Adam``). Everything that defines the
learning problem is kept identical on purpose, because any difference would
make a NumPy-vs-PyTorch comparison measure the recipe instead of the
framework:

* He-normal initialization, std = sqrt(2 / fan_in), zero biases.
* Loss = squared error summed over the outputs, averaged over the batch.
  This is the loss whose gradient the NumPy code writes as (2/m)(out - y).
  ``nn.MSELoss`` would also average over the outputs, and although Adam is
  insensitive to a constant gradient scale, the L2 term is not, so that
  would silently change the effective regularization.
* L2 added to the weight gradients only, not the biases, as in NumPy. That
  is coupled weight decay (``Adam(weight_decay=...)``), not ``AdamW``.
* Input dropout zeroes standardized channels (zero == the training mean)
  with no 1/(1-p) rescaling. ``nn.Dropout`` rescales, so it is not used.
* Standardization statistics from the training rows only, float64.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.ml.checks import (
    check_features,
    check_norm_stats,
    check_schema,
    check_targets,
    sensor_ids,
)
from src.ml.dataset import PARAM_NAMES
from src.ml.mlp import MLPRegressor

CHECKPOINT_FORMAT = "cute-surrogate-torch"
CHECKPOINT_VERSION = 1
_DTYPES = {"float64": torch.float64, "float32": torch.float32}


def resolve_device(device: str = "cpu") -> torch.device:
    """Turn "cpu", "cuda", "mps" or "auto" into a device, failing if absent."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' requested but Apple MPS is not available")
    return dev


def mask_inputs(xb: torch.Tensor, rate: float, generator: torch.Generator) -> torch.Tensor:
    """Zero each entry with probability ``rate``, without rescaling the rest.

    The mask is drawn on the CPU generator so a seeded run is reproducible
    regardless of which device the batch lives on.
    """
    keep = torch.rand(xb.shape, generator=generator, dtype=xb.dtype) >= rate
    return xb * keep.to(xb.device)


class MLPNet(nn.Module):
    """ReLU hidden layers and a linear output layer."""

    def __init__(self, n_in: int, n_out: int, hidden_layers: tuple[int, ...]):
        super().__init__()
        sizes = [n_in, *hidden_layers, n_out]
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])
        )

    @property
    def linears(self) -> list[nn.Linear]:
        return [layer for layer in self.layers if isinstance(layer, nn.Linear)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


@dataclass
class TorchMLPRegressor:
    """PyTorch counterpart of :class:`src.ml.mlp.MLPRegressor`.

    Args:
        hidden_layers, lr, batch_size, epochs, seed, l2, input_dropout: As in
            the NumPy model, with the same defaults.
        device: "cpu" (default), "cuda", "mps" or "auto".
        dtype: "float64" (default, matches NumPy) or "float32". Apple MPS
            only supports float32.
    """

    hidden_layers: tuple[int, ...] = (128, 128)
    lr: float = 3e-3
    batch_size: int = 128
    epochs: int = 300
    seed: int = 0
    l2: float = 1e-6
    input_dropout: float = 0.0
    device: str = "cpu"
    dtype: str = "float64"

    net: MLPNet | None = field(default=None, repr=False)
    x_mean_: np.ndarray | None = field(default=None, repr=False)
    x_std_: np.ndarray | None = field(default=None, repr=False)
    y_mean_: np.ndarray | None = field(default=None, repr=False)
    y_std_: np.ndarray | None = field(default=None, repr=False)
    feature_names_: list[str] = field(default_factory=list, repr=False)
    target_names_: list[str] = field(default_factory=list, repr=False)
    history_: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}; got {self.dtype!r}")
        if not 0.0 <= self.input_dropout < 1.0:
            raise ValueError(f"input_dropout must be in [0, 1); got {self.input_dropout}")
        self._device = resolve_device(self.device)
        if self._device.type == "mps" and self.dtype == "float64":
            raise ValueError("Apple MPS does not support float64; pass dtype='float32'")
        self._torch_dtype = _DTYPES[self.dtype]

    # -- internals -------------------------------------------------------------

    @property
    def n_features(self) -> int:
        if self.x_mean_ is None:
            raise RuntimeError("Model must be fit or loaded first.")
        return len(self.x_mean_)

    def _require_net(self) -> MLPNet:
        if self.net is None:
            raise RuntimeError("Model must be fit or loaded first.")
        return self.net

    def _init_net(self, n_in: int, n_out: int, generator: torch.Generator) -> None:
        net = MLPNet(n_in, n_out, self.hidden_layers).to(self._torch_dtype)
        with torch.no_grad():
            for layer in net.linears:
                std = float(np.sqrt(2.0 / layer.in_features))
                layer.weight.normal_(0.0, std, generator=generator)
                layer.bias.zero_()
        self.net = net.to(self._device)

    def _optimizer(self) -> torch.optim.Adam:
        net = self._require_net()
        return torch.optim.Adam(
            [
                {"params": [lyr.weight for lyr in net.linears], "weight_decay": self.l2},
                {"params": [lyr.bias for lyr in net.linears], "weight_decay": 0.0},
            ],
            lr=self.lr, betas=(0.9, 0.999), eps=1e-8,
        )

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=self._torch_dtype, device=self._device)

    def _standardize_x(self, X: np.ndarray) -> np.ndarray:
        assert self.x_mean_ is not None and self.x_std_ is not None
        return (X - self.x_mean_) / self.x_std_

    def _run_epochs(self, X: np.ndarray, y: np.ndarray, epochs: int,
                    generator: torch.Generator,
                    X_val: np.ndarray | None = None,
                    y_val: np.ndarray | None = None) -> None:
        """Train the current weights for ``epochs`` on already-validated data."""
        assert self.y_mean_ is not None and self.y_std_ is not None
        net = self._require_net()
        Xs = self._to_tensor(self._standardize_x(X))
        ys = self._to_tensor((y - self.y_mean_) / self.y_std_)
        opt = self._optimizer()
        n = Xs.shape[0]
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(n, generator=generator).to(self._device)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb = Xs[idx], ys[idx]
                if self.input_dropout > 0.0:
                    xb = mask_inputs(xb, self.input_dropout, generator)
                loss = ((net(xb) - yb) ** 2).sum(dim=1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if X_val is not None and y_val is not None:
                err = self.predict(X_val) - y_val
                self.history_.append(float(np.mean(err ** 2)))

    # -- public API ----------------------------------------------------------

    def fit(self, X, y, X_val=None, y_val=None,
            feature_names: list[str] | None = None,
            target_names: list[str] | None = None) -> "TorchMLPRegressor":
        """Train on (X, y). Standardizes inputs and targets from X and y only.

        ``X_val``/``y_val`` only feed ``history_``; they never change the
        weights. Feature and target names default to the project schema
        (sensor ids, :data:`PARAM_NAMES`) when the widths match it.
        """
        X_arr = np.asarray(X)
        X = check_features(X_arr, X_arr.shape[-1] if X_arr.ndim else 0)
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        y = check_targets(y_arr, y_arr.shape[1])
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} samples but y has {len(y)}")
        if X_val is not None and y_val is not None:
            X_val = check_features(X_val, X.shape[1], name="X_val")
            y_val = check_targets(np.asarray(y_val).reshape(len(X_val), -1),
                                  y.shape[1], name="y_val")

        self.feature_names_ = _names(feature_names, X.shape[1], sensor_ids(), "x")
        self.target_names_ = _names(target_names, y.shape[1], list(PARAM_NAMES), "y")
        self.x_mean_, self.x_std_ = X.mean(0), X.std(0) + 1e-12
        self.y_mean_, self.y_std_ = y.mean(0), y.std(0) + 1e-12
        self.history_ = []

        generator = torch.Generator().manual_seed(self.seed)
        self._init_net(X.shape[1], y.shape[1], generator)
        self._run_epochs(X, y, self.epochs, generator, X_val, y_val)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict targets in original units, as a float64 NumPy array."""
        net = self._require_net()
        assert self.y_mean_ is not None and self.y_std_ is not None
        X = check_features(X, self.n_features)
        net.eval()
        with torch.no_grad():
            out = net(self._to_tensor(self._standardize_x(X)))
        return out.cpu().double().numpy() * self.y_std_ + self.y_mean_

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save weights, normalization stats, hyperparameters and schema."""
        net = self._require_net()
        if self.x_mean_ is None or self.x_std_ is None \
                or self.y_mean_ is None or self.y_std_ is None:
            raise RuntimeError("Model must be fit before saving.")
        torch.save({
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "torch_version": str(torch.__version__),  # TorchVersion is not weights_only-safe
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "hidden_layers": list(self.hidden_layers),
            "hyperparameters": {
                "lr": self.lr, "batch_size": self.batch_size, "epochs": self.epochs,
                "seed": self.seed, "l2": self.l2, "input_dropout": self.input_dropout,
                "dtype": self.dtype,
            },
            "x_mean": torch.from_numpy(self.x_mean_),
            "x_std": torch.from_numpy(self.x_std_),
            "y_mean": torch.from_numpy(self.y_mean_),
            "y_std": torch.from_numpy(self.y_std_),
            "feature_names": list(self.feature_names_),
            "target_names": list(self.target_names_),
        }, str(path))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu",
             check_current_schema: bool = True) -> "TorchMLPRegressor":
        """Load a model saved with :meth:`save`.

        ``weights_only=True`` means the file is read as plain tensors and
        values, so loading an untrusted checkpoint cannot run code. With
        ``check_current_schema`` the saved sensor and target order must
        match the current sensor configuration and :data:`PARAM_NAMES`.
        """
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
        if ckpt.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"{path} has checkpoint version {ckpt.get('version')}; this code "
                f"reads version {CHECKPOINT_VERSION}"
            )
        hp = ckpt["hyperparameters"]
        model = cls(hidden_layers=tuple(ckpt["hidden_layers"]), device=device, **hp)
        model.x_mean_ = ckpt["x_mean"].numpy()
        model.x_std_ = ckpt["x_std"].numpy()
        model.y_mean_ = ckpt["y_mean"].numpy()
        model.y_std_ = ckpt["y_std"].numpy()
        model.feature_names_ = list(ckpt["feature_names"])
        model.target_names_ = list(ckpt["target_names"])
        n_in, n_out = len(model.x_mean_), len(model.y_mean_)
        check_norm_stats(model.x_mean_, model.x_std_, n_in, "input")
        check_norm_stats(model.y_mean_, model.y_std_, n_out, "target")
        if check_current_schema:
            check_schema(model.feature_names_, model.target_names_)

        net = MLPNet(n_in, n_out, model.hidden_layers).to(model._torch_dtype)
        net.load_state_dict(ckpt["state_dict"], strict=True)
        model.net = net.to(model._device)
        return model

    @classmethod
    def from_numpy(cls, reference: MLPRegressor, **kwargs) -> "TorchMLPRegressor":
        """Copy a fitted NumPy model's weights and statistics into PyTorch.

        With the same weights the two implementations must give the same
        predictions; that is the check that the forward passes agree.
        """
        if (reference.x_mean_ is None or reference.x_std_ is None
                or reference.y_mean_ is None or reference.y_std_ is None):
            raise RuntimeError("Reference model must be fit first.")
        hidden = tuple(W.shape[1] for W in reference.weights[:-1])
        model = cls(hidden_layers=hidden, lr=reference.lr,
                    batch_size=reference.batch_size, epochs=reference.epochs,
                    seed=reference.seed, l2=reference.l2,
                    input_dropout=reference.input_dropout, **kwargs)
        model.x_mean_, model.x_std_ = reference.x_mean_.copy(), reference.x_std_.copy()
        model.y_mean_, model.y_std_ = reference.y_mean_.copy(), reference.y_std_.copy()
        n_in, n_out = reference.weights[0].shape[0], reference.weights[-1].shape[1]
        model.feature_names_ = _names(None, n_in, sensor_ids(), "x")
        model.target_names_ = _names(None, n_out, list(PARAM_NAMES), "y")
        net = MLPNet(n_in, n_out, hidden).to(model._torch_dtype)
        with torch.no_grad():
            for layer, W, b in zip(net.linears, reference.weights, reference.biases):
                layer.weight.copy_(torch.from_numpy(W.T))  # nn.Linear stores (out, in)
                layer.bias.copy_(torch.from_numpy(b))
        model.net = net.to(model._device)
        return model


def _names(given: list[str] | None, n: int, schema: list[str], prefix: str) -> list[str]:
    if given is not None:
        if len(given) != n:
            raise ValueError(f"got {len(given)} names for {n} columns")
        return list(given)
    if len(schema) == n:
        return list(schema)
    return [f"{prefix}{i}" for i in range(n)]
