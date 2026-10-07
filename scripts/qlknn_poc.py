#!/usr/bin/env python
"""Proof of concept: feed a CUTE equilibrium into DeepMind's QLKNN_7_11.

What this is for
----------------
TORAX (Google DeepMind's open-source tokamak transport simulator) predicts how
a plasma's temperature and density evolve. Its default turbulence model is
QLKNN_7_11, a neural network in the ``fusion_surrogates`` library. This
script pipes this repo's output, a TokaMaker equilibrium of CUTE, into that
network, to find out what breaks. It is a roadblock finder, not a validated
transport prediction. See ``docs/qlknn_poc.md`` for the write-up.

Two stages, run separately, because the repo keeps its Open Fusion Toolkit
half and its ML half apart:

1. ``equilibrium`` (needs OFT): solve a CUTE equilibrium, extract the flux
   surface geometry and q profile to ``models/qlknn_poc_equilibrium.json``,
   and write the equilibrium as EQDSK files (the format TORAX reads) in both
   sign conventions TokaMaker supports.
2. ``qlknn`` (needs the ``qlknn`` extra): build QLKNN's ten inputs from that
   geometry plus ASSUMED temperature and density profiles, check them against
   the ranges QLKNN was trained on, run the network, and write
   ``models/qlknn_poc_results.json`` and ``docs/qlknn_poc.png``.

Usage
-----
    # Stage 1, with OFT on the path (see docs/operator_guide.md)
    PYTHONPATH=~/opt/oft/OpenFUSIONToolkit/python:$PWD \\
        python scripts/qlknn_poc.py equilibrium
    # Stage 2
    pip install -e ".[qlknn]"
    python scripts/qlknn_poc.py qlknn
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.coupling.qlknn_inputs import (  # noqa: E402
    EPSILON_NN,
    INPUT_NAMES,
    TRAINING_BOXES,
    FluxSurfaceGeometry,
    KineticAssumptions,
    combine_fluxes,
    compute_inputs,
    gyrobohm_diffusivity,
    heat_diffusivity,
    input_matrix,
    training_range_report,
)

EQUILIBRIUM_JSON = PROJECT_ROOT / "models" / "qlknn_poc_equilibrium.json"
RESULTS_JSON = PROJECT_ROOT / "models" / "qlknn_poc_results.json"
FIGURE = PROJECT_ROOT / "docs" / "qlknn_poc.png"

# The middle of the shape ranges in scripts/generate_gs_dataset.py, so this
# equilibrium is a typical member of the repo's Grad-Shafranov training set.
NOMINAL_SHAPE = {"ip": 1.0e5, "r0": 0.32, "z0": 0.0, "a": 0.165,
                 "kappa": 1.7, "delta": 0.4}


# --------------------------------------------------------------------------
# Stage 1: equilibrium (OFT)
# --------------------------------------------------------------------------
def stage_equilibrium(args: argparse.Namespace) -> None:
    from generate_gs_dataset import apply_shape, build_tokamaker, solve_is_sane

    mygs = build_tokamaker(PROJECT_ROOT / "data" / "CUTE_mesh.h5", nthreads=2)
    apply_shape(mygs, **NOMINAL_SHAPE)
    # OFT v26.9's solve() raises on failure, so the sanity check is the test.
    mygs.solve()
    stats = mygs.get_stats()
    if not solve_is_sane(stats, NOMINAL_SHAPE["ip"]):
        raise SystemExit(f"TokaMaker solve is not sane: {stats}")

    fsa = mygs.get_fsa(npsi=args.npsi, psi_pad=0.01)
    R_major = float(stats["R_geo"])
    geom = FluxSurfaceGeometry(
        psi_norm=np.asarray(fsa["psi_norm"]), q=np.abs(np.asarray(fsa["q"])),
        R_in=np.asarray(fsa["R_min"]), R_out=np.asarray(fsa["R_max"]),
        R_major=R_major, a_minor=float(stats["a_geo"]),
        B0=float(fsa["F0"]) / R_major,
    )

    eqdsk = {}
    for cocos in (7, 2):
        path = PROJECT_ROOT / "data" / f"qlknn_poc_cocos{cocos}.geqdsk"
        mygs.save_eqdsk(str(path), nr=args.eqdsk_grid, nz=args.eqdsk_grid, cocos=cocos,
                        run_info=f"CUTE qlknn_poc cocos={cocos}")
        eqdsk[f"cocos{cocos}"] = str(path.relative_to(PROJECT_ROOT))

    out = {
        "_about": ("Flux-surface geometry of one TokaMaker equilibrium of CUTE, "
                   "written by scripts/qlknn_poc.py. Synthetic: solved from "
                   "requested shape and current, not reconstructed from data. "
                   "The pressure and current profile shapes are the same "
                   "placeholders scripts/generate_gs_dataset.py uses."),
        "requested_shape": NOMINAL_SHAPE,
        "achieved": {k: float(stats[k]) for k in
                     ("Ip", "R_geo", "a_geo", "kappa", "delta", "q_95", "beta_pol", "l_i")
                     if k in stats},
        "diverted": bool(fsa["diverted"]),
        "geometry": geom.to_dict(),
        "eqdsk_files": eqdsk,
    }
    EQUILIBRIUM_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {EQUILIBRIUM_JSON.relative_to(PROJECT_ROOT)}")
    print(f"  Ip {stats['Ip']/1e3:.1f} kA, R {R_major:.3f} m, a {geom.a_minor:.3f} m, "
          f"a/R {geom.a_minor/R_major:.2f}, B0 {geom.B0:.3f} T, "
          f"q from {geom.q.min():.2f} to {geom.q.max():.2f}")
    for k, v in eqdsk.items():
        print(f"wrote {v}")


# --------------------------------------------------------------------------
# Stage 2: QLKNN (fusion_surrogates)
# --------------------------------------------------------------------------
def scan_assumptions(geom: FluxSurfaceGeometry) -> list[dict]:
    """Fraction of the plasma inside QLKNN's training data, across assumptions.

    The kinetic profiles are guesses, so the useful question is not "what does
    QLKNN predict for this one guess" but "for which plausible CUTE plasmas is
    QLKNN even being asked something it was trained on".
    """
    rows = []
    for Te0 in (0.05, 0.1, 0.2, 0.5, 1.0):
        for ne0 in (0.5e19, 1e19, 2e19, 5e19):
            k = KineticAssumptions(Te0_keV=Te0, Te_edge_keV=0.1 * Te0, ne0=ne0,
                                   ne_edge=0.2 * ne0)
            rep = training_range_report(compute_inputs(geom, k))
            core = rep["failures_by_box"]["QLKNN11D (core)"]
            worst = max(core, key=lambda n: core[n])
            rows.append({"Te0_keV": Te0, "ne0": ne0,
                         "fraction_inside": rep["fraction_inside"],
                         "most_common_failure": worst if core[worst] else None})
    return rows


def stage_qlknn(args: argparse.Namespace) -> None:
    from src.coupling import qlknn_runner

    eq = json.loads(EQUILIBRIUM_JSON.read_text())
    geom = FluxSurfaceGeometry.from_dict(eq["geometry"])
    k = KineticAssumptions()
    inputs = compute_inputs(geom, k)
    report = training_range_report(inputs)

    model = qlknn_runner.load_model()
    fluxes = qlknn_runner.predict_fluxes(model, input_matrix(inputs))
    total = combine_fluxes(fluxes)
    chi_gb = gyrobohm_diffusivity(inputs["Ti_keV"], geom.B0, k.A_i, geom.a_minor)
    chi_i = heat_diffusivity(total["qi"], inputs["Ati"], chi_gb, geom.R_major, geom.a_minor)
    chi_e = heat_diffusivity(total["qe"], inputs["Ate"], chi_gb, geom.R_major, geom.a_minor)

    eps_lcfs = geom.a_minor / geom.R_major
    rho_x_limit = TRAINING_BOXES["QLKNN11D (core)"]["x"][1] * EPSILON_NN / eps_lcfs
    inside = report["inside"]
    scan = scan_assumptions(geom)

    results = {
        "_about": ("Proof of concept output of scripts/qlknn_poc.py. Geometry "
                   "from a synthetic TokaMaker equilibrium; temperature and "
                   "density are ASSUMED placeholders, not CUTE measurements. "
                   "Fluxes are QLKNN_7_11 predictions in gyro-Bohm units; "
                   "points outside its training data are extrapolation."),
        "fusion_surrogates_version": qlknn_runner.PINNED_VERSION,
        "model": {"name": model.name, "version": model.version},
        "kinetic_assumptions": asdict(k),
        "geometry_summary": {"R_major": geom.R_major, "a_minor": geom.a_minor,
                             "a_over_R": eps_lcfs, "B0": geom.B0,
                             "qlknn_training_a_over_R": EPSILON_NN},
        "x_rescaling": {
            "rho_where_x_reaches_training_max": float(rho_x_limit),
            "explanation": ("TORAX rescales x by (a/R)/(1/3). For CUTE's a/R "
                            "the rescaled x passes QLKNN's training maximum of "
                            "0.95 at this normalized radius, so every surface "
                            "further out is outside the core training data on "
                            "geometry alone."),
        },
        "range_report": {
            "fraction_of_surfaces_inside": report["fraction_inside"],
            "failures_by_box": report["failures_by_box"],
        },
        "profiles": {
            "rho": geom.rho.tolist(),
            "psi_norm": geom.psi_norm.tolist(),
            "inside_training_data": inside.tolist(),
            **{n: np.asarray(inputs[n]).tolist() for n in
               INPUT_NAMES + ["x_unscaled", "alpha", "Te_keV", "Ti_keV", "ne"]},
            **{f"flux_{n}": v.tolist() for n, v in fluxes.items()},
            "qi_gB": total["qi"].tolist(), "qe_gB": total["qe"].tolist(),
            "chi_gB_m2s": chi_gb.tolist(),
            "chi_i_m2s": chi_i.tolist(), "chi_e_m2s": chi_e.tolist(),
        },
        "assumption_scan": scan,
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2))

    print(f"QLKNN {model.name} (fusion_surrogates {qlknn_runner.PINNED_VERSION})")
    print(f"CUTE a/R = {eps_lcfs:.2f} vs QLKNN training a/R = {EPSILON_NN:.2f}; "
          f"rescaled x exceeds 0.95 beyond r/a = {rho_x_limit:.2f}")
    print(f"surfaces inside QLKNN's training data: {inside.sum()} of {inside.size} "
          f"({100 * report['fraction_inside']:.0f}%)")
    print("inputs pushing surfaces outside the core box:")
    for name, n in report["failures_by_box"]["QLKNN11D (core)"].items():
        if n:
            v = np.asarray(inputs[name])
            lo, hi = TRAINING_BOXES["QLKNN11D (core)"][name]
            print(f"  {name:10s} {n:3d} surfaces   CUTE {v.min():8.3g} .. {v.max():8.3g}"
                  f"   trained {lo:g} .. {hi:g}")
    if inside.any():
        print(f"chi_i on in-range surfaces: {np.nanmin(chi_i[inside]):.2g} .. "
              f"{np.nanmax(chi_i[inside]):.2g} m^2/s; chi_e: "
              f"{np.nanmin(chi_e[inside]):.2g} .. {np.nanmax(chi_e[inside]):.2g} m^2/s")
    print("\nfraction inside training data across assumed profiles (rows Te0 keV, cols ne0):")
    ne_vals = sorted({r["ne0"] for r in scan})
    print("        " + "".join(f"{n:>9.1e}" for n in ne_vals))
    for Te0 in sorted({r["Te0_keV"] for r in scan}):
        row = [r for r in scan if r["Te0_keV"] == Te0]
        print(f"{Te0:6.2f}  " + "".join(f"{100 * r['fraction_inside']:8.0f}%" for r in row))
    print(f"\nwrote {RESULTS_JSON.relative_to(PROJECT_ROOT)}")
    if not args.no_figure:
        from qlknn_poc_figure import plot
        plot(results, FIGURE)
        print(f"wrote {FIGURE.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    p1 = sub.add_parser("equilibrium", help="solve with TokaMaker (needs OFT)")
    p1.add_argument("--npsi", type=int, default=60)
    p1.add_argument("--eqdsk-grid", type=int, default=129,
                    help="R and Z points in the EQDSK grid (TokaMaker alone defaults to 65)")
    p2 = sub.add_parser("qlknn", help="run QLKNN (needs the qlknn extra)")
    p2.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    {"equilibrium": stage_equilibrium, "qlknn": stage_qlknn}[args.stage](args)


if __name__ == "__main__":
    main()
