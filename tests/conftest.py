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


@pytest.fixture(scope="session")
def greens_session(tokamaker_session):
    """Session-scoped Green's function matrix.

    Computed once early before any test can corrupt the vacuum solver state.
    """
    from src.forward.sensors import generate_cute_sensors
    from src.reconstruct.greens import compute_greens_matrix

    mygs = tokamaker_session
    sensor_config = generate_cute_sensors()
    G, coil_names = compute_greens_matrix(mygs, sensor_config)
    return G, coil_names, sensor_config


@pytest.fixture(scope="session")
def vacuum_consistency_data(tokamaker_session, greens_session):
    """Pre-compute vacuum consistency check data before state corruption.

    The vacuum solver state is corrupted by eig_wall and non-vacuum solves,
    so all vacuum solve verification must happen early.
    """
    from src.reconstruct.constraints import forward_to_vector

    mygs = tokamaker_session
    G, coil_names, sensor_config = greens_session

    # Baseline (same as Green's matrix computation)
    baseline_currents = {cn: 0.0 for cn in coil_names}
    baseline_currents[coil_names[0]] = 1e-6
    mygs.set_coil_currents(baseline_currents)
    mygs.set_saddles(None)
    mygs.set_isoflux(None)
    mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
    mygs.solve(vacuum=True)
    y_baseline = forward_to_vector(mygs, sensor_config)

    # Actual vacuum solve with test currents
    test_currents = {name: 100.0 for name in coil_names}
    solve_currents = {name: 100.0 + baseline_currents.get(name, 0.0) for name in coil_names}
    mygs.set_coil_currents(solve_currents)
    mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
    mygs.solve(vacuum=True)
    y_actual = forward_to_vector(mygs, sensor_config)

    return y_baseline, y_actual, test_currents


@pytest.fixture(scope="session")
def eddy_session(tokamaker_session, greens_session, vacuum_consistency_data):
    """Session-scoped vessel response and TD data.

    Must be computed BEFORE any non-vacuum solve, since TokaMaker's vacuum
    solver state is corrupted by non-vacuum solves.
    """
    from src.forward.model import flux_loop, mirnov_probe
    from src.reconstruct.eddy import compute_vessel_response

    mygs = tokamaker_session
    _, _, sensor_config = greens_session

    # Compute vessel response (includes its own vacuum solves)
    vr = compute_vessel_response(
        mygs, sensor_config, step_coil="CS01", step_amplitude=100.0,
        n_modes=3, dt=1e-5, n_steps=80,
    )

    # Also compute TD data for fit quality tests (needs vacuum solves)
    coil_names = sorted(mygs.coil_sets.keys())
    step_amplitude = 100.0

    def _eval_sensors():
        psi_eval = mygs.get_field_eval("psi")
        B_eval = mygs.get_field_eval("B")
        vec = np.empty(sensor_config.n_total)
        for si, sensor in enumerate(sensor_config.flux_loops):
            vec[si] = flux_loop(psi_eval, (sensor["R"], sensor["Z"]))
        offset = sensor_config.n_flux_loops
        for si, sensor in enumerate(sensor_config.mirnov_probes):
            vec[offset + si] = mirnov_probe(
                B_eval, (sensor["R"], sensor["Z"]), sensor["angle"]
            )
        return vec

    c_pre = {cn: 0.0 for cn in coil_names}
    c_pre[coil_names[0]] = 1e-6
    c_post = dict(c_pre)
    c_post["CS01"] = step_amplitude

    # Steady state (vacuum solve BEFORE eig_wall)
    mygs.set_saddles(None)
    mygs.set_isoflux(None)
    mygs.set_coil_currents(c_post)
    mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
    mygs.solve(vacuum=True)
    y_ss = _eval_sensors()

    # Initial state
    mygs.set_coil_currents(c_pre)
    mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
    mygs.solve(vacuum=True)

    mygs.eig_wall(neigs=3, pm=False)
    mygs.setup_td(1e-5, 1e-8, 1e-8)
    mygs.set_coil_currents(c_post)

    times = []
    responses = []
    dt = 1e-5
    for i in range(80):
        t = (i + 1) * dt
        mygs.step_td(t, dt)
        responses.append(_eval_sensors())
        times.append(t)

    td_times = np.array(times)
    td_responses = np.array(responses)
    td_eddy = td_responses - y_ss[np.newaxis, :]

    return vr, sensor_config, td_times, td_eddy, step_amplitude


def _solve_reference(mygs):
    """Solve the reference CUTE equilibrium.

    Restores all necessary state (profiles, targets, constraints, coil settings)
    to ensure a clean solve regardless of what previous tests may have done.
    """
    from OpenFUSIONToolkit.TokaMaker.util import create_isoflux, create_power_flux_fun

    # Restore profiles (may have been changed by other solvers)
    ffp_prof = create_power_flux_fun(40, 1.5, 2.0)
    pp_prof = create_power_flux_fun(40, 4.0, 1.0)
    mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)

    # Restore coil settings
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

    mygs.set_targets(Ip=200.0e3, Ip_ratio=4.0)

    isoflux_pts = create_isoflux(80, 0.32, 0.0, 0.17, 1.7, 0.4)
    isoflux_pts = isoflux_pts[isoflux_pts[:, 0] > 0.3, :]
    isoflux_pts = np.vstack((isoflux_pts, np.array([[0.15, 0.0]])))
    x_points = np.array([[0.22, -0.33], [0.20, 0.34]])
    mygs.set_saddles(x_points)
    mygs.set_isoflux(np.vstack((isoflux_pts, x_points)))

    # Reset vacuum solver state (eig_wall/TD can corrupt it)
    mygs.vac_solve()

    mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
    mygs.solve()
