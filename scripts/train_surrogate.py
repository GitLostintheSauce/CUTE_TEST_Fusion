"""Train the ML equilibrium-reconstruction surrogate and write artifacts.

Outputs:
    models/surrogate.npz          trained network + normalization stats
    models/surrogate_metrics.json accuracy + speed benchmark
    docs/surrogate_benchmark.png  figure for the README / portfolio

Run:
    python scripts/train_surrogate.py --samples 8000 --epochs 400
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Train the CUTE ML surrogate")
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench-shots", type=int, default=60,
                        help="Shots used for the speed/accuracy benchmark")
    args = parser.parse_args()

    from src.ml.baseline import invert_least_squares
    from src.ml.dataset import PARAM_NAMES, PARAM_UNITS, generate_dataset
    from src.ml.surrogate import train_surrogate

    print(f"Training surrogate on {args.samples} samples, {args.epochs} epochs...")
    t0 = time.perf_counter()
    model, layout, metrics = train_surrogate(
        n_samples=args.samples, noise_frac=args.noise, seed=args.seed,
        epochs=args.epochs,
    )
    train_time = time.perf_counter() - t0
    print(f"  trained in {train_time:.1f}s   R2(overall) = {metrics.r2_overall:.4f}")

    # Held-out benchmark set (unseen seed).
    Xb, yb, _ = generate_dataset(n_samples=args.bench_shots, noise_frac=args.noise,
                                 seed=args.seed + 777, layout=layout)

    # Surrogate inference timing (per shot, amortized over a batch).
    t0 = time.perf_counter()
    y_sur = model.predict(Xb)
    t_sur = (time.perf_counter() - t0) / len(Xb)

    # Classical iterative baseline timing + accuracy.
    y_base = np.empty_like(yb)
    t0 = time.perf_counter()
    for i in range(len(Xb)):
        y_base[i] = invert_least_squares(Xb[i], layout)
    t_base = (time.perf_counter() - t0) / len(Xb)

    speedup = t_base / t_sur
    print(f"  surrogate {t_sur * 1e6:.1f} us/shot   "
          f"baseline {t_base * 1e3:.2f} ms/shot   speedup {speedup:.0f}x")

    # Save model + metrics.
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model.save(str(models_dir / "surrogate.npz"))

    out = metrics.as_dict()
    out.update({
        "train_time_s": train_time,
        "surrogate_us_per_shot": t_sur * 1e6,
        "baseline_ms_per_shot": t_base * 1e3,
        "speedup": speedup,
        "noise_frac": args.noise,
    })
    with open(models_dir / "surrogate_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved model + metrics to {models_dir}")

    # Benchmark figure.
    try:
        _make_figure(yb, y_sur, y_base, PARAM_NAMES, PARAM_UNITS,
                     t_sur, t_base, PROJECT_ROOT / "docs" / "surrogate_benchmark.png")
        print("  wrote docs/surrogate_benchmark.png")
    except Exception as exc:  # plotting is optional, never fail the run on it
        print(f"  (skipped figure: {exc})")


def _make_figure(y_true, y_sur, y_base, names, units, t_sur, t_base, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4 * (len(names) + 1), 4))
    for j, (name, unit) in enumerate(zip(names, units)):
        ax = axes[j]
        ax.scatter(y_true[:, j], y_sur[:, j], s=16, alpha=0.7,
                   color="#0072B2", label="surrogate")
        lo, hi = y_true[:, j].min(), y_true[:, j].max()
        ax.plot([lo, hi], [lo, hi], "--", color="#7f8c99", lw=1)
        ax.set_xlabel(f"true {name} ({unit})")
        ax.set_ylabel(f"predicted {name} ({unit})")
        ax.set_title(name)

    ax = axes[-1]
    ax.bar(["surrogate", "iterative\nbaseline"],
           [t_sur * 1e6, t_base * 1e6],
           color=["#0072B2", "#E69F00"])
    ax.set_yscale("log")
    ax.set_ylabel("microseconds per shot (log)")
    ax.set_title(f"Inference speed ({t_base / t_sur:.0f}x faster)")

    fig.suptitle("ML surrogate vs. iterative reconstruction", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
