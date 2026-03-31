"""Phase 2 acceptance tests: TokaMaker reference equilibrium."""
import os

import numpy as np
import pytest

# Skip all tests in this module if OFT is not available
oft_available = True
try:
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.util import create_isoflux, create_power_flux_fun
except ImportError:
    oft_available = False

pytestmark = pytest.mark.skipif(not oft_available, reason="OFT not installed")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def cute_equilibrium():
    """Create and return a converged CUTE reference equilibrium.

    This fixture is module-scoped so the expensive solve only runs once.
    """
    myOFT = OFT_env(nthreads=2)
    mygs = TokaMaker(myOFT)

    mesh_path = os.path.join(PROJECT_ROOT, "data", "CUTE_mesh.h5")
    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(mesh_path)
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
    mygs.settings.lim_zmax = 0.38
    mygs.settings.pm = False
    mygs.setup(order=2, F0=0.17)

    # Coil configuration
    coil_bounds = {key: [-1.0e3, 1.0e3] for key in mygs.coil_sets}
    mygs.set_coil_bounds(coil_bounds)

    coil_mirrors = {"CS{0:02d}".format(2 * i + 1): "CS{0:02d}".format(2 * i + 2) for i in range(7)}
    coil_mirrors.update({"PF{0:02d}".format(i): "PF{0:02d}".format(15 - i) for i in range(1, 8)})
    disable_list = ["PF01"]

    regularization_terms = []
    for name in mygs.coil_sets:
        if name not in coil_mirrors:
            continue
        if name in disable_list:
            regularization_terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.0e5))
            regularization_terms.append(
                mygs.coil_reg_term({coil_mirrors[name]: 1.0}, target=0.0, weight=1.0e5)
            )
        else:
            regularization_terms.append(
                mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.0e-1)
            )
            regularization_terms.append(
                mygs.coil_reg_term(
                    {name: 1.0, coil_mirrors[name]: -1.0}, target=0.0, weight=1.0e0
                )
            )
    mygs.set_coil_reg(reg_terms=regularization_terms)

    ffp_prof = create_power_flux_fun(40, 1.5, 2.0)
    pp_prof = create_power_flux_fun(40, 4.0, 1.0)
    mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)

    Ip_target = 200.0e3
    beta_approx = 0.2
    mygs.set_targets(Ip=Ip_target, Ip_ratio=(1.0 / beta_approx - 1.0))

    isoflux_pts = create_isoflux(80, 0.32, 0.0, 0.17, 1.7, 0.4)
    isoflux_pts = isoflux_pts[isoflux_pts[:, 0] > 0.3, :]
    isoflux_pts = np.vstack((isoflux_pts, np.array([[0.15, 0.0]])))
    x_points = np.array([[0.22, -0.33], [0.20, 0.34]])
    mygs.set_saddles(x_points)
    mygs.set_isoflux(np.vstack((isoflux_pts, x_points)))

    mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
    mygs.solve()

    return mygs


def test_static_solve_produces_valid_equilibrium(cute_equilibrium):
    """[2.4] Static solve produces valid equilibrium: Ip > 0, flux surfaces exist."""
    mygs = cute_equilibrium
    stats = mygs.get_stats()

    assert stats["Ip"] > 0, f"Ip should be positive, got {stats['Ip']}"
    assert stats["kappa"] > 1.0, f"Elongation should be > 1, got {stats['kappa']}"

    # Check that o_point is in a physically reasonable location
    assert 0.2 < mygs.o_point[0] < 0.5, f"O-point R={mygs.o_point[0]} outside reasonable range"
    assert abs(mygs.o_point[1]) < 0.1, f"O-point Z={mygs.o_point[1]} too far from midplane"

    # Verify psi has a reasonable range (not flat)
    psi = mygs.get_psi(False)
    assert np.ptp(psi) > 0, "Psi field is flat — no equilibrium"


def test_time_dependent_solve_completes(cute_equilibrium):
    """[2.5] Time-dependent solve runs to completion with > 10 time steps."""
    mygs = cute_equilibrium

    # Clear shape constraints for free evolution
    mygs.set_saddles(None)
    mygs.set_isoflux(None)

    dt = 1.0e-4
    mygs.setup_td(dt, 1.0e-13, 1.0e-11)

    sim_time = 0.0
    n_steps = 15
    times = []
    for i in range(n_steps):
        sim_time, _, nl_its, lin_its, nretry = mygs.step_td(sim_time, dt)
        assert nretry >= 0, f"Time step {i} failed with nretry={nretry}"
        times.append(sim_time)

    assert len(times) > 10, f"Expected > 10 time steps, got {len(times)}"
    assert times[-1] > times[0], "Time should advance"


def test_bfield_extraction(cute_equilibrium):
    """[2.6] B-field extraction at arbitrary points gives physically reasonable values."""
    mygs = cute_equilibrium

    # Re-solve to get clean static equilibrium (TD may have perturbed)
    isoflux_pts = create_isoflux(80, 0.32, 0.0, 0.17, 1.7, 0.4)
    isoflux_pts = isoflux_pts[isoflux_pts[:, 0] > 0.3, :]
    isoflux_pts = np.vstack((isoflux_pts, np.array([[0.15, 0.0]])))
    x_points = np.array([[0.22, -0.33], [0.20, 0.34]])
    mygs.set_saddles(x_points)
    mygs.set_isoflux(np.vstack((isoflux_pts, x_points)))
    mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
    mygs.solve()

    B_eval = mygs.get_field_eval("B")
    test_points = np.array([
        [0.32, 0.0],
        [0.40, 0.1],
        [0.20, -0.2],
    ])

    for pt in test_points:
        B_vals = B_eval.eval(pt)
        assert not np.any(np.isnan(B_vals)), f"NaN B-field at ({pt[0]}, {pt[1]})"
        B_mag = np.sqrt(B_vals[0] ** 2 + B_vals[1] ** 2 + B_vals[2] ** 2)
        assert 0 < B_mag < 2.0, f"|B|={B_mag} T outside range (0, 2) at ({pt[0]}, {pt[1]})"
