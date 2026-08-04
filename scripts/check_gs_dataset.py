#!/usr/bin/env python
"""Check whether a Grad-Shafranov dataset supports non-circular labels.

The reduced dataset in ``src/ml/dataset.py`` computes q95, beta_pol and l_i
from Ip by formula, so a network predicting them would only be relearning that
formula. That is why the shipped surrogate deliberately predicts Ip, R0, Z0 and
a, and nothing else.

Whether a Grad-Shafranov dataset removes that objection is an empirical
question, not an assumption: these quantities *can* still correlate strongly
with Ip through the physics. This script measures it, so the claim rests on a
number rather than on the fact that a real solver was used.

Reported per quantity:

* Pearson r against Ip. Near +/-1 means it is still essentially a restatement
  of Ip and predicting it would remain close to circular.
* The R2 of a least-squares fit on Ip alone. This is the share of the variance
  that Ip already explains; the remainder is what a network could add.

Usage
-----
    python scripts/check_gs_dataset.py --data data/gs_dataset.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def r_squared_against(x: np.ndarray, y: np.ndarray) -> float:
    """R2 of the best straight-line fit y ~ x."""
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/gs_dataset.npz")
    args = parser.parse_args()

    path = PROJECT_ROOT / args.data
    if not path.exists():
        print(f"Dataset not found: {path}. Run scripts/generate_gs_dataset.py "
              "first.", file=sys.stderr)
        return 1

    d = np.load(path, allow_pickle=True)
    y = d["y"]
    extras = d["extras"]
    param_names = [str(s) for s in d["param_names"]]
    extra_names = [str(s) for s in d["extra_names"]]
    ip = y[:, param_names.index("Ip")]

    print(f"{path.name}: {y.shape[0]} equilibria from "
          f"{str(d['source'])}")
    if "psi" in d:
        print(f"psi maps present: {d['psi'].shape[0]} x {d['psi'].shape[1]} nodes")
    print()
    print("Dependence on Ip (lower is better: more genuinely new information)")
    print(f"  {'quantity':>10}  {'pearson r':>10}  {'R2 vs Ip':>9}  verdict")

    for j, name in enumerate(extra_names):
        col = extras[:, j]
        mask = np.isfinite(col)
        if mask.sum() < 10 or np.std(col[mask]) == 0:
            print(f"  {name:>10}  {'n/a':>10}  {'n/a':>9}  constant or missing")
            continue
        r = float(np.corrcoef(ip[mask], col[mask])[0, 1])
        r2 = r_squared_against(ip[mask], col[mask])
        if r2 > 0.95:
            verdict = "still essentially Ip, do not predict"
        elif r2 > 0.7:
            verdict = "largely explained by Ip, weak target"
        else:
            verdict = "carries independent information"
        print(f"  {name:>10}  {r:>10.3f}  {r2:>9.3f}  {verdict}")

    print()
    print("For contrast, the same measure on the labels the surrogate already "
          "predicts:")
    for j, name in enumerate(param_names):
        if name == "Ip":
            continue
        col = y[:, j]
        r2 = r_squared_against(ip, col)
        print(f"  {name:>10}  {float(np.corrcoef(ip, col)[0, 1]):>10.3f}  "
              f"{r2:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
