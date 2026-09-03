#!/usr/bin/env python
"""Reconstruct a CUTE equilibrium with TokaMaker's own reconstruction module.

Why this exists
---------------
The reconstruction in ``src/reconstruct/efit.py`` is hand-rolled: a Green's
function matrix plus a Tikhonov-regularized least-squares decomposition of the
measured field into coil and plasma contributions. It was written without
reference to ``OpenFUSIONToolkit.TokaMaker.reconstruction``, which is the
toolkit's own equilibrium reconstruction, and which already defines the
constraint types it needs (flux loops, Mirnov probes, plasma current,
saddles, pressure, q).

This script uses the toolkit's version instead, so the comparison between the
two is a measurement rather than an assumption.

It also fixes a structural problem in the hand-rolled path. ``efit.py`` sets
isoflux and saddle targets to a hardcoded shape and then fits, so the plasma
boundary is an input to the reconstruction rather than an output of it.
TokaMaker's ``reconstruct()`` refuses that combination: it strips isoflux and
saddle targets and warns, because shape targets are *design* constraints
("give me this boundary") and diagnostics are *measurement* constraints
("match these readings"). Both determine the boundary, so they conflict.

What it does
------------
1. Solves a truth equilibrium with design constraints (isoflux plus saddles).
2. Evaluates the 130 magnetic diagnostics on it, optionally with noise. These
   become the measurements.
3. Clears the design constraints, perturbs the solver's starting point, and
   reconstructs from the measurements alone.
4. Compares the reconstructed equilibrium against the truth it never saw.

Step 3 is what keeps this from being an inverse crime in the strict sense: the
truth is produced by shape targets, and the reconstruction runs with those
targets removed, driven only by sensor values. It is still the same solver and
the same mesh on both sides, which is a weaker guarantee than reconstructing
someone else's equilibrium, and that limit is reported rather than hidden.

On fitting coil currents
------------------------
The default settings hold coil currents fixed (``fitCoils = False``) and free
only the profile scale factors ``Pnorm`` (the scale on P', the pressure
gradient) and ``alam`` (the scale on FF', the poloidal current function). That
is physically right rather than a shortcut: on a real device the coil currents
are measured directly, so they are known inputs. What reconstruction actually
infers is the plasma contribution. ``--fit-coils`` frees them anyway, which is
the harder and more ill-conditioned problem.

What running this found
-----------------------
Three results worth keeping, all reproducible with the commands below.

1. **The reference equilibrium cannot be reconstructed, because it is not a
   free-boundary equilibrium.** The reference configuration (Ip = 200 kA,
   a = 0.17, kappa = 1.7, delta = 0.4) solves only while the isoflux targets
   hold it there. Remove them, keep the same coil currents, and the solve
   exceeds its iteration limit. Reconstruction needs an equilibrium that
   exists for the given coil currents, so there is nothing to reconstruct.
   Removing the X-points alone does not fix it; the shape has to be gentler.
   Note also that 10 to 13 of the 28 coils sit exactly on their 1 kA bound in
   every configuration tried, so the reference shape is being bought at the
   limit of what the coil set can do.

2. **A viable configuration reconstructs well.** At Ip = 100 kA, a = 0.15,
   kappa = 1.5, delta = 0.2 with no X-points, the fit converges and recovers
   every compared quantity to under 1%, including q_95, beta_pol and l_i.
   Those three are the ones the ML surrogate deliberately excludes as
   circular, and here they are genuine outputs of a Grad-Shafranov solve.
   Adding 2% sensor noise barely moves the result, because 131 constraints
   against 2 free parameters averages the noise out.

3. **Freeing the coil currents makes the fit degenerate.** With --fit-coils
   there are 30 free parameters instead of 2. The fit still reports success
   (error flag 0) and still gets the boundary about right (R_geo and a_geo
   within 0.1%), but the plasma is nonsense: Ip comes back as 2.8 kA against a
   truth of 100 kA, beta_pol as 743 against 28, l_i as 1690 against 1.2. Many
   different coil-current and plasma-pressure combinations reproduce the same
   external field, so the external magnetics alone cannot separate them. This
   is why the defaults hold the coils fixed, and it is worth knowing that a
   converged fit is not automatically a correct one.

A note on the API
-----------------
``add_Mirnov``'s docstring says the normal is "in the R-Z plane [2]", but
``Mirnov_con.write()`` writes three components, so passing a 2-vector raises
``IndexError: list index out of range``. The normal is a 3-vector and the
docstring is wrong. The component ordering is undocumented, hence
``--norm-order``; the ``rpz`` default is accepted by the Fortran reader.

Usage
-----
    # The reference configuration, which fails for reason 1 above
    python scripts/reconstruct_with_oft.py

    # A configuration that reconstructs. VIABLE holds these values.
    python scripts/reconstruct_with_oft.py --ip 100e3 --a 0.15 --kappa 1.5 --delta 0.2
    python scripts/reconstruct_with_oft.py --ip 100e3 --a 0.15 --kappa 1.5 \
        --delta 0.2 --noise-frac 0.02
    python scripts/reconstruct_with_oft.py --ip 100e3 --a 0.15 --kappa 1.5 \
        --delta 0.2 --fit-coils
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forward.model import full_diagnostic_set  # noqa: E402
from src.forward.sensors import generate_cute_sensors  # noqa: E402

# The truth equilibrium. Same configuration as the reference solve in
# notebooks/cute_reference_equilibrium.py so the two are comparable.
TRUTH = {
    "Ip": 200.0e3,
    "R0": 0.32,
    "Z0": 0.0,
    "a": 0.17,
    "kappa": 1.7,
    "delta": 0.4,
}

# A configuration that does support reconstruction, found by trial. Unlike
# TRUTH, this one survives as a free-boundary equilibrium once the shape
# targets are removed. See "What running this found" above.
VIABLE = {"Ip": 1.0e5, "R0": 0.32, "Z0": 0.0, "a": 0.15, "kappa": 1.5, "delta": 0.2}

# Quantities compared between truth and reconstruction. Ip is the plasma
# current, R_geo and a_geo the geometric major and minor radius, q_95 the
# safety factor near the edge, beta_pol the poloidal beta, l_i the internal
# inductance.
COMPARE = ["Ip", "R_geo", "a_geo", "kappa", "delta", "q_95", "beta_pol", "l_i"]


def build_tokamaker(mesh_path: Path, nthreads: int):
    """Set up TokaMaker on the CUTE mesh.

    Mirrors src/reconstruct/cli.py and scripts/generate_gs_dataset.py so the
    equilibria here match the rest of the pipeline.
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
    mygs.set_coil_bounds({key: [-1.0e3, 1.0e3] for key in mygs.coil_sets})
    return mygs


def solve_truth(mygs, truth: dict, use_xpoints: bool):
    """Solve the truth equilibrium using shape targets, the design direction."""
    from OpenFUSIONToolkit.TokaMaker.util import create_isoflux, create_power_flux_fun

    mygs.set_profiles(ffp_prof=create_power_flux_fun(40, 1.5, 2.0),
                      pp_prof=create_power_flux_fun(40, 4.0, 1.0))

    r0, z0 = truth["R0"], truth["Z0"]
    a, kappa, delta = truth["a"], truth["kappa"], truth["delta"]

    isoflux_pts = create_isoflux(80, r0, z0, a, kappa, delta)
    isoflux_pts = isoflux_pts[isoflux_pts[:, 0] > r0 - a + 0.02, :]

    if use_xpoints:
        x_points = np.array([[r0 - 0.10, z0 - 0.33], [r0 - 0.12, z0 + 0.34]])
        mygs.set_saddles(x_points)
        mygs.set_isoflux(np.vstack((isoflux_pts, x_points)))
    else:
        mygs.set_saddles(None)
        mygs.set_isoflux(isoflux_pts)

    mygs.set_targets(Ip=truth["Ip"], Ip_ratio=4.0)
    mygs.init_psi(r0, z0, a * 0.75, kappa, delta)
    mygs.solve()
    return mygs.get_stats()


def measure(mygs, sensors, noise_frac: float, rng) -> tuple[dict, dict]:
    """Evaluate the 130 diagnostics, optionally with noise.

    Returns the measured values and the per-sensor error used to weight the
    fit. Errors are set from the noise level with a floor at 0.5% of the
    family RMS, so a sensor that happens to read near zero does not get an
    error of zero and dominate the fit.
    """
    frame = full_diagnostic_set(mygs, sensors)
    ids = [c for c in frame.columns if c != "time"]
    clean = {sid: float(frame[sid].iloc[0]) for sid in ids}

    fl_ids = [s["id"] for s in sensors.flux_loops]
    mp_ids = [s["id"] for s in sensors.mirnov_probes]

    values, errors = {}, {}
    for family in (fl_ids, mp_ids):
        rms = float(np.sqrt(np.mean([clean[s] ** 2 for s in family])))
        floor = 0.005 * rms
        for sid in family:
            v = clean[sid]
            sigma = max(noise_frac * abs(v), floor)
            values[sid] = v + (rng.normal(0.0, sigma) if noise_frac > 0 else 0.0)
            errors[sid] = sigma if sigma > 0 else 1.0
    return values, errors


def build_reconstruction(mygs, sensors, values, errors, ip_truth, args):
    """Register the diagnostics as constraints on a reconstruction object.

    This is the pattern from the toolkit's own reconstruction example: create
    the reconstruction object first, then define each diagnostic in terms of
    it. Every constraint carries an error, so the fit is weighted by
    measurement uncertainty rather than treating all channels equally.
    """
    from OpenFUSIONToolkit.TokaMaker.reconstruction import reconstruction

    recon = reconstruction(mygs)

    for s in sensors.flux_loops:
        recon.add_flux_loop([s["R"], s["Z"]], values[s["id"]], errors[s["id"]])

    for s in sensors.mirnov_probes:
        # add_Mirnov's docstring says the normal is "in the R-Z plane [2]", but
        # Mirnov_con.write() writes three components and Mirnov_con.read()
        # reads three, so a 2-vector raises IndexError. The normal is a full 3D
        # unit vector and the docstring is wrong.
        #
        # The component ordering is not documented anywhere in the Python
        # layer. The location line is written as (R, Z, phi), which would
        # suggest (n_R, n_Z, n_phi), but a 3D field normal is more naturally
        # (n_R, n_phi, n_Z). Only the Fortran reader settles it, so both are
        # available here and the difference is measured rather than guessed.
        #
        # src/forward/model.py projects the field as
        # Br*cos(angle) + Bz*sin(angle), so the probe has no toroidal
        # sensitivity and the toroidal component is zero either way.
        c, s_ = float(np.cos(s["angle"])), float(np.sin(s["angle"]))
        norm = [c, 0.0, s_] if args.norm_order == "rpz" else [c, s_, 0.0]
        recon.add_Mirnov([s["R"], s["Z"]], norm, values[s["id"]], errors[s["id"]])

    # Plasma current is a measured quantity on a real machine (Rogowski coil),
    # so it enters as a constraint with an error, not as a design target.
    recon.set_Ip(ip_truth, args.ip_err * abs(ip_truth))

    recon.settings.fitPnorm = True
    recon.settings.fitAlam = True
    recon.settings.fitCoils = args.fit_coils
    recon.settings.fitR0 = args.fit_r0
    recon.settings.fitV0 = args.fit_r0
    recon.settings.pm = args.verbose
    return recon


def report(truth_stats: dict, recon_stats: dict) -> None:
    print()
    print("Truth vs reconstruction")
    print(f"  {'quantity':>9}  {'truth':>12}  {'recon':>12}  {'rel err':>9}")
    for key in COMPARE:
        t = truth_stats.get(key)
        r = recon_stats.get(key)
        if t is None or r is None:
            print(f"  {key:>9}  {'n/a':>12}  {'n/a':>12}  {'n/a':>9}")
            continue
        t, r = float(t), float(r)
        rel = abs(r - t) / abs(t) if t != 0 else float("nan")
        print(f"  {key:>9}  {t:>12.5g}  {r:>12.5g}  {rel:>8.2%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=str, default="data/CUTE_mesh.h5")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-frac", type=float, default=0.0,
                        help="fractional sensor noise, also sets constraint errors")
    parser.add_argument("--ip-err", type=float, default=0.02,
                        help="fractional error on the plasma current constraint")
    parser.add_argument("--fit-coils", action="store_true",
                        help="free the coil currents during the fit "
                             "(harder and more ill-conditioned; on a real "
                             "machine the coil currents are measured)")
    parser.add_argument("--fit-r0", action="store_true",
                        help="free the R0 and Z0 centering constraints")
    parser.add_argument("--linearized", action="store_true",
                        help="use the linearized solve for suitable terms")
    parser.add_argument("--ip", type=float, default=TRUTH["Ip"])
    parser.add_argument("--r0", type=float, default=TRUTH["R0"])
    parser.add_argument("--z0", type=float, default=TRUTH["Z0"])
    parser.add_argument("--a", type=float, default=TRUTH["a"])
    parser.add_argument("--kappa", type=float, default=TRUTH["kappa"])
    parser.add_argument("--delta", type=float, default=TRUTH["delta"])
    parser.add_argument("--x-points", action="store_true",
                        help="add saddle constraints and pin the boundary at "
                             "the X-points, as the reference equilibrium does")
    parser.add_argument("--perturb", type=float, default=1.0,
                        help="how far from the truth shape to start the fit; "
                             "0 starts at the truth initial guess")
    parser.add_argument("--clear-targets", action="store_true",
                        help="clear the Ip design target before reconstructing")
    parser.add_argument("--norm-order", choices=["rpz", "rzp"], default="rpz",
                        help="component order of the Mirnov normal vector: "
                             "rpz = (n_R, n_phi, n_Z), rzp = (n_R, n_Z, n_phi). "
                             "Undocumented in OFT, so both are provided")
    parser.add_argument("--maxits", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mesh_path = PROJECT_ROOT / args.mesh
    if not mesh_path.exists():
        print(f"Mesh not found: {mesh_path}", file=sys.stderr)
        return 1

    truth = {"Ip": args.ip, "R0": args.r0, "Z0": args.z0,
             "a": args.a, "kappa": args.kappa, "delta": args.delta}

    rng = np.random.default_rng(args.seed)
    sensors = generate_cute_sensors()

    print(f"Setting up TokaMaker on {mesh_path.name} ...")
    mygs = build_tokamaker(mesh_path, args.threads)

    print("Solving the truth equilibrium (design direction, shape targets) ...")
    truth_stats = solve_truth(mygs, truth, args.x_points)
    ip_truth = float(truth_stats["Ip"])
    print(f"  Ip = {ip_truth / 1e3:.1f} kA, "
          f"R_geo = {float(truth_stats['R_geo']):.4f} m, "
          f"a_geo = {float(truth_stats['a_geo']):.4f} m")

    print(f"Evaluating {sensors.n_total} diagnostics "
          f"({sensors.n_flux_loops} flux loops, "
          f"{sensors.n_mirnov_probes} Mirnov probes), "
          f"noise = {args.noise_frac:.1%} ...")
    values, errors = measure(mygs, sensors, args.noise_frac, rng)

    print("Building the reconstruction and registering constraints ...")
    recon = build_reconstruction(mygs, sensors, values, errors, ip_truth, args)

    # Clear the design constraints explicitly. reconstruct() would strip these
    # itself and warn, but doing it here makes the distinction visible: the
    # shape targets built the truth, and the reconstruction must not see them.
    mygs.set_isoflux(None)
    mygs.set_saddles(None)

    if args.clear_targets:
        # The Ip target is a design constraint too. The fit gets its plasma
        # current from the Ip constraint registered above, so the target may be
        # redundant or in conflict.
        mygs.set_targets(Ip=-1.0, Ip_ratio=-1.0)

    # Start the non-linear solve away from the answer, so the reconstruction is
    # not simply sitting on the truth when it begins. --perturb 0 starts at the
    # truth shape, which isolates whether a failure is about the starting point
    # or about the fit itself.
    p = args.perturb
    mygs.init_psi(truth["R0"] + 0.01 * p, truth["Z0"],
                  truth["a"] * 0.75 - 0.02 * p,
                  truth["kappa"] - 0.2 * p, truth["delta"] - 0.1 * p)

    print(f"Reconstructing (fitCoils={args.fit_coils}, "
          f"fitPnorm=True, fitAlam=True, "
          f"linearized={args.linearized}) ...")
    try:
        err_flag = recon.reconstruct(linearized_fit=args.linearized,
                                     maxits=args.maxits)
    except Exception:
        print("\nReconstruction raised:", file=sys.stderr)
        traceback.print_exc()
        print("\nThat traceback is the useful output. Record it.",
              file=sys.stderr)
        return 2

    print(f"  error flag: {err_flag}")

    try:
        recon_stats = mygs.get_stats()
    except Exception:
        print("\nget_stats() failed after reconstruction, which usually means "
              "the flux surface tracing could not find a clean boundary.",
              file=sys.stderr)
        traceback.print_exc()
        return 3

    report(truth_stats, recon_stats)

    print()
    print("Honest limits of this comparison:")
    print("  - Same solver, same mesh, and the same profile shapes on both")
    print("    sides. Only the profile scale factors and the starting point")
    print("    differ, so this is weaker than reconstructing an equilibrium")
    print("    produced by someone else's code.")
    print("  - The sensor layout in src/forward/sensors.py is invented, not")
    print("    CUTE's real diagnostic set, so these errors describe a")
    print("    plausible machine rather than the actual one.")
    if not args.fit_coils:
        print("  - Coil currents were held at their truth values, which is")
        print("    realistic (they are measured) but means this does not test")
        print("    recovering them. Use --fit-coils for that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
