#!/usr/bin/env python
"""Build a labeled dataset from real Grad-Shafranov solves.

Why this exists
---------------
The surrogate in ``src/ml/`` is trained on the *reduced* forward model in
``src/ml/dataset.py``: a rigid disk of circular current filaments evaluated
analytically. That model is honest and fast, but it is not a free-boundary
Grad-Shafranov solve, so the surrogate's benchmark has to be a least-squares
inversion of the *same* reduced model. Comparing it against TokaMaker would
mostly measure "reduced physics vs full physics" rather than "network vs
classical inversion", which would inflate the number and mean less.

This script removes that limitation at the source. It drives TokaMaker across a
range of plasma states, evaluates the 130 magnetic diagnostics on each solved
equilibrium with :mod:`src.forward.model`, and records the equilibrium's own
scalar parameters as labels. A surrogate trained on the output is learning real
Grad-Shafranov equilibria, so it can be compared against TokaMaker directly.

It also produces two things the reduced model cannot:

* **Real psi maps** (``--save-psi``), which retire the "illustrative" label on
  the dashboard's flux contours (roadmap 3.1).
* **Genuinely independent q95, beta_pol and l_i.** In the reduced dataset these
  are computed from Ip by formula, so predicting them would be circular. Here
  they are measured from the solved equilibrium (roadmap 3.2).

Cost: roughly 0.26 s per solve on the CUTE mesh, so a few thousand samples is
tens of minutes. Not all requested shapes converge; the yield is reported.

Usage
-----
    python scripts/generate_gs_dataset.py --samples 2000 --out data/gs_dataset.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forward.model import full_diagnostic_set  # noqa: E402
from src.forward.sensors import generate_cute_sensors  # noqa: E402

# Sampling ranges for the requested plasma shape. These are the *targets* handed
# to TokaMaker; the achieved equilibrium is measured afterwards and is what gets
# stored as the label. Ranges are deliberately narrower than the reduced model's
# because a free-boundary solve has to actually be achievable with CUTE's coils.
SHAPE_RANGES = {
    "Ip": (4.0e4, 1.8e5),    # plasma current target [A]
    "R0": (0.30, 0.34),      # major radius [m]
    "Z0": (-0.03, 0.03),     # vertical position [m]
    "a": (0.14, 0.19),       # minor radius [m]
    "kappa": (1.5, 1.9),     # elongation
    "delta": (0.3, 0.5),     # triangularity
}

# Labels, matching the reduced-model surrogate's output order.
PARAM_NAMES = ["Ip", "R0", "Z0", "a"]

# Additional measured quantities. Unlike in the reduced dataset these are
# independent properties of the solved equilibrium, not formulas applied to Ip.
EXTRA_NAMES = ["kappa", "delta", "q_95", "beta_pol", "l_i", "vol", "W_MHD"]


def build_tokamaker(mesh_path: Path, nthreads: int):
    """Set up a TokaMaker instance on the CUTE mesh.

    Mirrors the configuration in src/reconstruct/cli.py so the equilibria here
    match the ones the reconstruction pipeline works with.
    """
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh

    myOFT = OFT_env(nthreads=nthreads)
    mygs = TokaMaker(myOFT)

    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(str(mesh_path))
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
    mygs.settings.lim_zmax = 0.38
    mygs.settings.pm = False
    mygs.setup(order=2, F0=0.17)

    # CUTE coils are limited to 1 kA per turn.
    mygs.set_coil_bounds({key: [-1.0e3, 1.0e3] for key in mygs.coil_sets})
    return mygs


def apply_shape(mygs, ip: float, r0: float, z0: float, a: float,
                kappa: float, delta: float) -> None:
    """Point TokaMaker at one requested plasma shape and current."""
    from OpenFUSIONToolkit.TokaMaker.util import create_isoflux, create_power_flux_fun

    mygs.set_profiles(ffp_prof=create_power_flux_fun(40, 1.5, 2.0),
                      pp_prof=create_power_flux_fun(40, 4.0, 1.0))

    isoflux_pts = create_isoflux(80, r0, z0, a, kappa, delta)
    # Drop inboard points that fall inside the central column, then pin the
    # inner edge, matching the constraint set used in src/reconstruct/solver.py.
    isoflux_pts = isoflux_pts[isoflux_pts[:, 0] > r0 - a + 0.02, :]
    isoflux_pts = np.vstack((isoflux_pts, np.array([[r0 - a - 0.02, z0]])))

    x_points = np.array([[r0 - 0.10, z0 - 0.33], [r0 - 0.12, z0 + 0.34]])
    mygs.set_saddles(x_points)
    mygs.set_isoflux(np.vstack((isoflux_pts, x_points)))

    mygs.set_targets(Ip=ip, Ip_ratio=4.0)
    mygs.init_psi(r0, z0, a * 0.75, kappa, delta)


def solve_is_sane(stats: dict, ip_target: float, tol: float = 0.25) -> bool:
    """Reject solves that converged to something unusable.

    A returned solution is not automatically a good one: the solver can settle
    far from the requested current or produce a degenerate cross section. Those
    samples would teach the surrogate a relationship that does not hold.
    """
    ip = stats.get("Ip")
    a_geo = stats.get("a_geo")
    r_geo = stats.get("R_geo")
    if ip is None or a_geo is None or r_geo is None:
        return False
    if not np.isfinite([ip, a_geo, r_geo]).all():
        return False
    if abs(ip - ip_target) > tol * abs(ip_target):
        return False
    if not (0.05 < a_geo < 0.25):
        return False
    if not (0.20 < r_geo < 0.50):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500,
                        help="number of plasma states to attempt")
    parser.add_argument("--out", type=str, default="data/gs_dataset.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--mesh", type=str, default="data/CUTE_mesh.h5")
    parser.add_argument("--save-psi", action="store_true",
                        help="also store the nodal psi map for each solve "
                             "(large; enables roadmap 3.1)")
    args = parser.parse_args()

    mesh_path = PROJECT_ROOT / args.mesh
    if not mesh_path.exists():
        print(f"Mesh not found: {mesh_path}", file=sys.stderr)
        return 1

    sensors = generate_cute_sensors()
    rng = np.random.default_rng(args.seed)

    print(f"Setting up TokaMaker on {mesh_path.name} ...")
    t_setup = time.perf_counter()
    mygs = build_tokamaker(mesh_path, args.threads)
    print(f"  setup took {time.perf_counter() - t_setup:.1f} s")

    X_rows: list[np.ndarray] = []
    y_rows: list[list[float]] = []
    extra_rows: list[list[float]] = []
    psi_rows: list[np.ndarray] = []
    sensor_ids: list[str] = []

    attempted = 0
    failed = 0
    solve_times: list[float] = []
    t_start = time.perf_counter()

    while len(y_rows) < args.samples:
        attempted += 1
        if attempted > args.samples * 4:
            print("Giving up: too many failed solves relative to the target.",
                  file=sys.stderr)
            break

        req = {k: float(rng.uniform(*v)) for k, v in SHAPE_RANGES.items()}
        try:
            apply_shape(mygs, req["Ip"], req["R0"], req["Z0"], req["a"],
                        req["kappa"], req["delta"])
            t0 = time.perf_counter()
            mygs.solve()
            solve_times.append(time.perf_counter() - t0)
            stats = mygs.get_stats()
        except Exception:
            failed += 1
            continue

        if not solve_is_sane(stats, req["Ip"]):
            failed += 1
            continue

        try:
            frame = full_diagnostic_set(mygs, sensors)
        except Exception:
            failed += 1
            continue

        if not sensor_ids:
            sensor_ids = [c for c in frame.columns if c != "time"]
        signals = frame[sensor_ids].to_numpy(dtype=float)[0]
        if not np.isfinite(signals).all():
            failed += 1
            continue

        # Labels come from the achieved equilibrium, never from the request.
        centroid = stats.get("Ip_centroid", [stats["R_geo"], 0.0])
        X_rows.append(signals)
        y_rows.append([float(stats["Ip"]), float(stats["R_geo"]),
                       float(centroid[1]), float(stats["a_geo"])])
        extra_rows.append([float(stats.get(k, np.nan)) for k in EXTRA_NAMES])

        if args.save_psi:
            psi_rows.append(np.asarray(mygs.get_psi(normalized=False),
                                       dtype=np.float32))

        n = len(y_rows)
        if n % 25 == 0 or n == args.samples:
            rate = (time.perf_counter() - t_start) / n
            remaining = (args.samples - n) * rate
            print(f"  {n}/{args.samples} kept  "
                  f"({failed} rejected)  "
                  f"~{remaining / 60:.1f} min left")

    if not y_rows:
        print("No usable equilibria produced.", file=sys.stderr)
        return 1

    X = np.array(X_rows)
    y = np.array(y_rows)
    extras = np.array(extra_rows)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "X": X,
        "y": y,
        "extras": extras,
        "sensor_ids": np.array(sensor_ids),
        "param_names": np.array(PARAM_NAMES),
        "extra_names": np.array(EXTRA_NAMES),
        "source": np.array("tokamaker_grad_shafranov"),
    }
    if psi_rows:
        payload["psi"] = np.array(psi_rows)
    np.savez_compressed(out_path, **payload)

    elapsed = time.perf_counter() - t_start
    median_solve = float(np.median(solve_times)) if solve_times else float("nan")
    print()
    print(f"Wrote {out_path}")
    print(f"  kept {len(y_rows)} of {attempted} attempts "
          f"({100 * len(y_rows) / attempted:.0f}% yield)")
    print(f"  {X.shape[1]} sensor channels, {y.shape[1]} labels, "
          f"{extras.shape[1]} measured extras")
    print(f"  median GS solve {median_solve * 1e3:.0f} ms, "
          f"total {elapsed / 60:.1f} min")
    if psi_rows:
        print(f"  psi maps: {len(psi_rows)} x {psi_rows[0].shape[0]} nodes")
    print()
    print("Label ranges (from the achieved equilibria, not the requests):")
    for j, name in enumerate(PARAM_NAMES):
        print(f"  {name:>3}: {y[:, j].min():12.4f} .. {y[:, j].max():12.4f}")
    print("Measured extras:")
    for j, name in enumerate(EXTRA_NAMES):
        col = extras[:, j]
        if np.isfinite(col).any():
            print(f"  {name:>9}: {np.nanmin(col):10.4f} .. {np.nanmax(col):10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
