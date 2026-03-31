"""Shared test fixtures for CUTE pipeline.

IMPORTANT: TokaMaker only allows ONE instance per Python kernel.
All tests that need TokaMaker must share the same instance via these fixtures.
"""
import os

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

oft_available = True
try:
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
except ImportError:
    oft_available = False


@pytest.fixture(scope="session")
def tokamaker_session():
    """Session-scoped TokaMaker instance with CUTE equilibrium.

    Only one TokaMaker can exist per process — this fixture ensures that.
    """
    if not oft_available:
        pytest.skip("OFT not installed")

    myOFT = OFT_env(nthreads=2)
    mygs = TokaMaker(myOFT)

    mesh_path = os.path.join(PROJECT_ROOT, "data", "CUTE_mesh.h5")
    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(mesh_path)
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
    mygs.settings.lim_zmax = 0.38
    mygs.settings.pm = False
    mygs.setup(order=2, F0=0.17)

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

    return mygs


def _solve_reference(mygs):
    """Solve the reference CUTE equilibrium."""
    from OpenFUSIONToolkit.TokaMaker.util import create_isoflux

    mygs.set_targets(Ip=200.0e3, Ip_ratio=4.0)

    isoflux_pts = create_isoflux(80, 0.32, 0.0, 0.17, 1.7, 0.4)
    isoflux_pts = isoflux_pts[isoflux_pts[:, 0] > 0.3, :]
    isoflux_pts = np.vstack((isoflux_pts, np.array([[0.15, 0.0]])))
    x_points = np.array([[0.22, -0.33], [0.20, 0.34]])
    mygs.set_saddles(x_points)
    mygs.set_isoflux(np.vstack((isoflux_pts, x_points)))

    mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
    mygs.solve()
