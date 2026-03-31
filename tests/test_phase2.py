"""Phase 2 acceptance tests: TokaMaker reference equilibrium."""
import numpy as np
import pytest

oft_available = True
try:
    import OpenFUSIONToolkit  # noqa: F401
except ImportError:
    oft_available = False

pytestmark = pytest.mark.skipif(not oft_available, reason="OFT not installed")


def test_static_solve_produces_valid_equilibrium(tokamaker_session):
    """[2.4] Static solve produces valid equilibrium: Ip > 0, flux surfaces exist."""
    from tests.conftest import _solve_reference

    mygs = tokamaker_session
    _solve_reference(mygs)
    stats = mygs.get_stats()

    assert stats["Ip"] > 0, f"Ip should be positive, got {stats['Ip']}"
    assert stats["kappa"] > 1.0, f"Elongation should be > 1, got {stats['kappa']}"
    assert 0.2 < mygs.o_point[0] < 0.5, f"O-point R={mygs.o_point[0]} outside range"
    assert abs(mygs.o_point[1]) < 0.1, f"O-point Z={mygs.o_point[1]} too far from midplane"

    psi = mygs.get_psi(False)
    assert np.ptp(psi) > 0, "Psi field is flat"


def test_time_dependent_solve_completes(tokamaker_session):
    """[2.5] Time-dependent solve runs > 10 steps."""
    from tests.conftest import _solve_reference

    mygs = tokamaker_session
    _solve_reference(mygs)

    mygs.set_saddles(None)
    mygs.set_isoflux(None)

    dt = 1.0e-4
    mygs.setup_td(dt, 1.0e-13, 1.0e-11)

    sim_time = 0.0
    n_steps = 15
    times = []
    for i in range(n_steps):
        sim_time, _, nl_its, lin_its, nretry = mygs.step_td(sim_time, dt)
        assert nretry >= 0, f"Time step {i} failed"
        times.append(sim_time)

    assert len(times) > 10
    assert times[-1] > times[0]


def test_bfield_extraction(tokamaker_session):
    """[2.6] B-field extraction at arbitrary points gives reasonable values."""
    from tests.conftest import _solve_reference

    mygs = tokamaker_session
    _solve_reference(mygs)

    B_eval = mygs.get_field_eval("B")
    test_points = np.array([[0.32, 0.0], [0.40, 0.1], [0.20, -0.2]])

    for pt in test_points:
        B_vals = B_eval.eval(pt)
        assert not np.any(np.isnan(B_vals)), f"NaN B-field at ({pt[0]}, {pt[1]})"
        B_mag = np.sqrt(B_vals[0] ** 2 + B_vals[1] ** 2 + B_vals[2] ** 2)
        assert 0 < B_mag < 2.0, f"|B|={B_mag} T outside range at ({pt[0]}, {pt[1]})"
