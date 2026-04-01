"""CLI entry point for CUTE equilibrium reconstruction."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="CUTE equilibrium reconstruction")
    parser.add_argument("--shot", required=True, help="Path to shot HDF5 file")
    parser.add_argument("--output", required=True, help="Path to output HDF5 file")
    parser.add_argument("--max-iter", type=int, default=50, help="Max iterations per slice")
    parser.add_argument("--tol", type=float, default=1e-8, help="Convergence tolerance")
    args = parser.parse_args()

    # Lazy imports to avoid loading OFT unless actually running
    try:
        from OpenFUSIONToolkit import OFT_env
        from OpenFUSIONToolkit.TokaMaker import TokaMaker
        from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
        from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
    except ImportError:
        print("Error: OpenFUSIONToolkit is not installed.", file=sys.stderr)
        sys.exit(1)

    from src.forward.sensors import generate_cute_sensors
    from src.reconstruct.solver import fit_equilibrium
    from src.store.hdf5 import load_shot, save_shot

    # Load shot data
    shot = load_shot(args.shot)
    print(f"Loaded shot {shot.metadata.shot_number}")

    # Set up TokaMaker
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mesh_path = os.path.join(project_root, "data", "CUTE_mesh.h5")

    myOFT = OFT_env(nthreads=2)
    mygs = TokaMaker(myOFT)

    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(mesh_path)
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
    mygs.settings.lim_zmax = 0.38
    mygs.settings.pm = False
    mygs.setup(order=2, F0=0.17)

    # Configure coils (same as reference)
    coil_bounds = {key: [-1.0e3, 1.0e3] for key in mygs.coil_sets}
    mygs.set_coil_bounds(coil_bounds)

    coil_mirrors = {
        "CS{0:02d}".format(2 * i + 1): "CS{0:02d}".format(2 * i + 2) for i in range(7)
    }
    coil_mirrors.update(
        {"PF{0:02d}".format(i): "PF{0:02d}".format(15 - i) for i in range(1, 8)}
    )
    disable_list = ["PF01"]

    reg_terms = []
    for name in mygs.coil_sets:
        if name not in coil_mirrors:
            continue
        if name in disable_list:
            reg_terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.0e5))
            reg_terms.append(
                mygs.coil_reg_term({coil_mirrors[name]: 1.0}, target=0.0, weight=1.0e5)
            )
        else:
            reg_terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.0e-1))
            reg_terms.append(
                mygs.coil_reg_term(
                    {name: 1.0, coil_mirrors[name]: -1.0}, target=0.0, weight=1.0e0
                )
            )
    mygs.set_coil_reg(reg_terms=reg_terms)

    ffp_prof = create_power_flux_fun(40, 1.5, 2.0)
    pp_prof = create_power_flux_fun(40, 4.0, 1.0)
    mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)

    # Extract measurements from processed signals
    sensor_config = generate_cute_sensors()
    signals = shot.processed or shot.raw
    if not signals:
        print("Error: No signal data in shot file.", file=sys.stderr)
        sys.exit(1)

    # For each time slice, extract sensor values
    # For now, use the mean value of each channel as a single time-slice measurement
    measurements = {}
    for ch_id, sig_data in signals.items():
        measurements[ch_id] = float(np.mean(sig_data.values))

    print(f"Reconstructing with {len(measurements)} measurements...")

    result = fit_equilibrium(
        mygs,
        measurements,
        sensor_config,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    print(f"Reconstruction complete: Ip={result.equilibrium.plasma_current:.0f} A, "
          f"chi²={result.diagnostics.chi_squared:.2e}, "
          f"iterations={result.diagnostics.n_iterations}")

    # Save result
    save_shot(
        args.output,
        shot.metadata,
        raw_signals=shot.raw,
        processed_signals=shot.processed,
        equilibrium=[result.equilibrium],
    )
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
