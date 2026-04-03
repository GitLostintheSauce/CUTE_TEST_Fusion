"""Phase 8a acceptance tests: EFIT-style reconstruction."""
import numpy as np
import pytest

from src.forward.sensors import generate_cute_sensors
from src.validation.benchmarks import noise_sweep

oft_available = True
try:
    import OpenFUSIONToolkit  # noqa: F401
except ImportError:
    oft_available = False

pytestmark = pytest.mark.skipif(not oft_available, reason="OFT not installed")


def _generate_clean_measurements(mygs, sensor_config):
    """Generate clean synthetic measurements from current equilibrium."""
    from src.forward.model import full_diagnostic_set

    df = full_diagnostic_set(mygs, sensor_config, time_index=0.0)
    return {col: float(df[col].iloc[0]) for col in df.columns if col != "time"}


@pytest.fixture(scope="module")
def efit_state(tokamaker_session, greens_session, eddy_session):
    """Set up Green's matrix and reference measurements for EFIT tests.

    Depends on eddy_session to ensure vacuum-dependent computations
    happen before non-vacuum solves corrupt the state.
    """
    from tests.conftest import _solve_reference

    mygs = tokamaker_session
    G, coil_names, sensor_config = greens_session

    # Solve reference equilibrium and generate measurements
    _solve_reference(mygs)
    stats = mygs.get_stats()
    clean_meas = _generate_clean_measurements(mygs, sensor_config)

    return mygs, G, coil_names, sensor_config, stats, clean_meas


def test_vacuum_coil_recovery(efit_state):
    """[8a.4] Vacuum-only reconstruction recovers coil currents within 1%."""
    _, G, coil_names, sensor_config, _, _ = efit_state

    # Generate vacuum-only measurements with known coil currents
    rng = np.random.default_rng(42)
    I_true = rng.uniform(-500, 500, len(coil_names))
    y_vacuum = G @ I_true

    # Recover coil currents via least-squares (no regularization for clean data)
    I_recovered = np.linalg.lstsq(G, y_vacuum, rcond=None)[0]

    # Check recovery
    for j, name in enumerate(coil_names):
        if abs(I_true[j]) > 1.0:  # skip near-zero currents
            rel_err = abs(I_recovered[j] - I_true[j]) / abs(I_true[j])
            assert rel_err < 0.01, (
                f"Coil {name}: {rel_err:.4f} > 1% (true={I_true[j]:.1f}, "
                f"recovered={I_recovered[j]:.1f})"
            )


def test_efit_zero_noise_flux_roundtrip(efit_state):
    """[8a.5] EFIT zero-noise: flux loop error < 0.5% of signal range."""
    from src.reconstruct.efit import efit_reconstruct
    from tests.conftest import _solve_reference

    mygs, G, coil_names, sensor_config, stats, clean_meas = efit_state
    _solve_reference(mygs)

    result = efit_reconstruct(
        mygs, clean_meas, sensor_config, G, coil_names, max_iter=10
    )

    # Check flux loop residuals
    fl_ids = [s["id"] for s in sensor_config.flux_loops]
    fl_meas = [clean_meas[sid] for sid in fl_ids]
    fl_range = max(fl_meas) - min(fl_meas) if len(fl_meas) > 1 else 1.0

    for sid in fl_ids:
        err = abs(result.diagnostics.per_sensor_residual[sid])
        assert err < 0.005 * abs(fl_range), (
            f"{sid}: error {err:.2e} > 0.5% of range {fl_range:.2e}"
        )


def test_efit_zero_noise_ip(efit_state):
    """[8a.6] EFIT zero-noise: Ip recovered within 2%."""
    from src.reconstruct.efit import efit_reconstruct
    from tests.conftest import _solve_reference

    mygs, G, coil_names, sensor_config, stats, clean_meas = efit_state
    _solve_reference(mygs)

    result = efit_reconstruct(
        mygs, clean_meas, sensor_config, G, coil_names, max_iter=10
    )

    true_ip = stats["Ip"]
    recon_ip = result.equilibrium.plasma_current
    rel_err = abs(recon_ip - true_ip) / abs(true_ip)
    assert rel_err < 0.02, f"Ip error {rel_err:.4f} > 2%"


def test_efit_zero_noise_boundary(efit_state):
    """[8a.7] EFIT zero-noise: boundary error < 1.0 cm."""
    from src.reconstruct.efit import efit_reconstruct
    from src.reconstruct.solver import _get_boundary_rz
    from tests.conftest import _solve_reference

    mygs, G, coil_names, sensor_config, stats, clean_meas = efit_state

    # Get ground truth boundary
    _solve_reference(mygs)
    true_r, true_z = _get_boundary_rz(mygs)

    # Reconstruct
    result = efit_reconstruct(
        mygs, clean_meas, sensor_config, G, coil_names, max_iter=10
    )

    recon_r = np.array(result.equilibrium.boundary_r)
    recon_z = np.array(result.equilibrium.boundary_z)
    true_r = np.array(true_r)
    true_z = np.array(true_z)

    max_dist = 0.0
    for i in range(len(recon_r)):
        dists = np.sqrt((true_r - recon_r[i]) ** 2 + (true_z - recon_z[i]) ** 2)
        max_dist = max(max_dist, float(np.min(dists)))

    assert max_dist < 0.01, f"Max boundary error {max_dist * 100:.2f} cm > 1.0 cm"


def test_efit_noisy_degradation(efit_state):
    """[8a.8] EFIT noisy: error increases monotonically, Ip error < 10% at SNR=10dB."""
    from src.reconstruct.efit import efit_reconstruct
    from tests.conftest import _solve_reference

    mygs, G, coil_names, sensor_config, stats, clean_meas = efit_state

    snr_levels = [float("inf"), 20.0, 10.0]
    noisy_sets = noise_sweep(clean_meas, snr_levels)

    chi_sq = {}
    ip_errors = {}
    for snr_db in snr_levels:
        _solve_reference(mygs)
        result = efit_reconstruct(
            mygs, noisy_sets[snr_db], sensor_config, G, coil_names, max_iter=10
        )
        chi_sq[snr_db] = result.diagnostics.chi_squared
        ip_errors[snr_db] = abs(
            result.equilibrium.plasma_current - stats["Ip"]
        ) / abs(stats["Ip"])

    # Monotonic chi-squared increase
    assert chi_sq[10.0] >= chi_sq[20.0], (
        f"10dB chi² ({chi_sq[10.0]:.2e}) < 20dB chi² ({chi_sq[20.0]:.2e})"
    )
    assert chi_sq[20.0] >= chi_sq[float("inf")], (
        f"20dB chi² ({chi_sq[20.0]:.2e}) < inf chi² ({chi_sq[float('inf')]:.2e})"
    )

    # Ip error bounded at SNR=10dB
    assert ip_errors[10.0] < 0.10, f"Ip error at SNR=10dB: {ip_errors[10.0]:.4f} > 10%"


def test_efit_convergence(efit_state):
    """[8a.9] EFIT converges in < 30 iterations."""
    from src.reconstruct.efit import efit_reconstruct
    from tests.conftest import _solve_reference

    mygs, G, coil_names, sensor_config, stats, clean_meas = efit_state
    _solve_reference(mygs)

    result = efit_reconstruct(
        mygs, clean_meas, sensor_config, G, coil_names, max_iter=30
    )

    assert result.diagnostics.n_iterations < 30, (
        f"EFIT did not converge in 30 iters (took {result.diagnostics.n_iterations})"
    )


def test_regularization(efit_state):
    """[8a.10] Regularized G'G + λI has condition number < 1e6."""
    from src.reconstruct.efit import select_regularization

    _, G, coil_names, sensor_config, _, clean_meas = efit_state
    from src.reconstruct.constraints import measurements_to_vector

    y = measurements_to_vector(clean_meas, sensor_config)
    lam = select_regularization(G, y)

    GtG = G.T @ G
    reg = GtG + lam * np.eye(len(coil_names))
    cond = np.linalg.cond(reg)
    assert cond < 1e6, f"Condition number {cond:.2e} > 1e6"


def test_efit_diagnostics(efit_state):
    """[8a.11] EFIT produces complete diagnostics."""
    from src.reconstruct.efit import efit_reconstruct
    from tests.conftest import _solve_reference

    mygs, G, coil_names, sensor_config, stats, clean_meas = efit_state
    _solve_reference(mygs)

    result = efit_reconstruct(
        mygs, clean_meas, sensor_config, G, coil_names, max_iter=10
    )

    diag = result.diagnostics
    assert isinstance(diag.chi_squared, float)
    assert diag.chi_squared >= 0
    assert len(diag.per_sensor_residual) == 130
    assert isinstance(diag.condition_number, float)
    assert diag.condition_number > 0
    assert len(result.coil_currents) == 28
    assert diag.n_iterations > 0
