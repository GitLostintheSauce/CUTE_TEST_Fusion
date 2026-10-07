#!/usr/bin/env python
"""Probe: can TORAX load this repo's TokaMaker equilibrium of CUTE?

TORAX does not take QLKNN inputs directly; it builds them from a geometry it
loads itself. The usual route from TokaMaker is an EQDSK file, a decades-old
text format for one equilibrium. The format does not record its own sign
conventions (which way the current flows, whether flux is divided by 2 pi, and
so on, collectively called COCOS), so the reader has to be told. TokaMaker
writes COCOS 7 by default and can also write COCOS 2.

This script loads both files written by ``scripts/qlknn_poc.py equilibrium``
through TORAX's EQDSK geometry loader, then compares what TORAX derives (major
and minor radius, field, q profile) against what TokaMaker reported. It also
loads the COCOS 7 file while wrongly declaring it COCOS 2, to see whether a
mismatch fails loudly or silently.

TORAX pins its own versions of several libraries (h5py, numpy-adjacent
packages), so run this in a separate environment rather than the repo's
``.venv``::

    python3.12 -m venv .venv-torax
    .venv-torax/bin/pip install "torax==1.4.3"
    .venv-torax/bin/python scripts/torax_geometry_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EQUILIBRIUM_JSON = PROJECT_ROOT / "models" / "qlknn_poc_equilibrium.json"
RESULTS_JSON = PROJECT_ROOT / "models" / "torax_geometry_probe.json"


def load(path: Path, cocos: int):
    from torax._src.geometry import eqdsk

    cfg = eqdsk.EQDSKConfig(geometry_file=path.name,
                            geometry_directory=str(path.parent),
                            cocos=cocos, n_rho=25)
    return cfg.build_geometry()


def summarize(geo, tm: dict) -> dict:
    """TORAX-derived quantities next to TokaMaker's, for one loaded geometry."""
    rho_mid = 0.5 * (np.asarray(geo.R_out_face) - np.asarray(geo.R_in_face))
    rho_mid = rho_mid / rho_mid[-1]
    psi = np.asarray(geo.psi_from_Ip_face)
    phi = np.asarray(geo.Phi_face)
    q_torax = np.abs(np.gradient(phi, psi))   # q = dPhi/dpsi in COCOS 11

    g = tm["geometry"]
    r_tm = 0.5 * (np.asarray(g["R_out"]) - np.asarray(g["R_in"])) / g["a_minor"]
    q_tm = np.interp(rho_mid, r_tm, np.asarray(g["q"]))
    inner = (rho_mid > 0.15) & (rho_mid < 0.85)
    rel = np.abs(q_torax[inner] - q_tm[inner]) / q_tm[inner]
    return {
        "R_major": float(geo.R_major), "a_minor": float(geo.a_minor),
        "B_0": float(geo.B_0),
        "q_relative_difference_0.15_to_0.85": {
            "median": float(np.median(rel)), "max": float(rel.max())},
        "q_torax_sample": [float(v) for v in q_torax[::5]],
        "q_tokamaker_sample": [float(v) for v in q_tm[::5]],
    }


def main() -> None:
    tm = json.loads(EQUILIBRIUM_JSON.read_text())
    ref = {"R_major": tm["geometry"]["R_major"], "a_minor": tm["geometry"]["a_minor"],
           "B0": tm["geometry"]["B0"]}
    print(f"TokaMaker: R {ref['R_major']:.4f} m, a {ref['a_minor']:.4f} m, B0 {ref['B0']:.4f} T")

    out: dict = {"tokamaker": ref, "cases": {}}
    cases = [("cocos7 file, declared 7", "qlknn_poc_cocos7.geqdsk", 7),
             ("cocos2 file, declared 2", "qlknn_poc_cocos2.geqdsk", 2),
             ("cocos7 file, WRONGLY declared 2", "qlknn_poc_cocos7.geqdsk", 2)]
    for label, fname, cocos in cases:
        try:
            s = summarize(load(PROJECT_ROOT / "data" / fname, cocos), tm)
            out["cases"][label] = {"loaded": True, **s}
            d = s["q_relative_difference_0.15_to_0.85"]
            print(f"{label:34s} loaded: R {s['R_major']:.4f}, a {s['a_minor']:.4f}, "
                  f"B0 {s['B_0']:.4f}; q vs TokaMaker median {100 * d['median']:.1f}% "
                  f"(max {100 * d['max']:.1f}%)")
        except Exception as e:  # noqa: BLE001 - the point is to record any failure
            msg = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
            out["cases"][label] = {"loaded": False, "error": msg}
            print(f"{label:34s} FAILED: {msg}")
    RESULTS_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {RESULTS_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
