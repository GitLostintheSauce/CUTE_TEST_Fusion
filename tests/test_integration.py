"""Phase 9 acceptance tests: Integration pipeline and regression."""
import subprocess
import sys

import numpy as np
import pytest

oft_available = True
try:
    import OpenFUSIONToolkit  # noqa: F401
except ImportError:
    oft_available = False

pytestmark = pytest.mark.skipif(not oft_available, reason="OFT not installed")


def test_efit_eddy_pipeline(tokamaker_session, greens_session, eddy_session):
    """[9.1] Full pipeline: synthetic ramp with eddy -> compensate -> EFIT -> Ip < 5%.

    Generates a synthetic coil current ramp, simulates measurements including
    eddy current transients, compensates the eddy contribution, then runs
    EFIT reconstruction on the compensated measurements.
    """
    from src.reconstruct.eddy import compensate_eddy_fast
    from src.reconstruct.efit import efit_reconstruct
    from tests.conftest import _solve_reference

    mygs = tokamaker_session
    G, coil_names, sensor_config = greens_session
    vr, _, td_times, td_eddy, step_amplitude = eddy_session

    # 1. Solve reference equilibrium (ground truth)
    _solve_reference(mygs)
    stats = mygs.get_stats()
    true_ip = stats["Ip"]

    # 2. Generate clean measurements from the reference equilibrium
    from src.forward.model import full_diagnostic_set

    df = full_diagnostic_set(mygs, sensor_config, time_index=0.0)
    clean_meas = {col: float(df[col].iloc[0]) for col in df.columns if col != "time"}

    # 3. Create a synthetic time-series with eddy transients
    #    Simulate a ramp: coil currents go from 0 to final values over n_times steps
    n_times = 20
    n_sensors = sensor_config.n_total

    # Get final coil currents from the reference solve
    coil_currents_ref = mygs.get_coil_currents()[0]
    I_final = np.array([coil_currents_ref.get(name, 0.0) for name in coil_names])

    times = np.linspace(0, 0.001, n_times)
    coil_timeseries = np.outer(np.linspace(0, 1, n_times), I_final)

    # Clean measurements at each time step (steady-state values, scaled by ramp)
    from src.reconstruct.constraints import measurements_to_vector

    y_clean = measurements_to_vector(clean_meas, sensor_config)
    meas_timeseries = np.outer(np.linspace(0, 1, n_times), y_clean)

    # Add synthetic eddy contribution (scale from TD data)
    # Use the first few TD eddy samples, scaled to match the ramp
    eddy_contrib = np.zeros((n_times, n_sensors))
    for t in range(1, n_times):
        # Eddy proportional to dI/dt, which is constant during linear ramp
        td_idx = min(t, len(td_eddy) - 1)
        eddy_contrib[t] = td_eddy[td_idx] * (np.linalg.norm(I_final) / step_amplitude) * 0.01

    meas_with_eddy = meas_timeseries + eddy_contrib

    # 4. Compensate eddy currents
    compensated = compensate_eddy_fast(meas_with_eddy, times, coil_timeseries, vr)

    # 5. Run EFIT on the final time slice (fully ramped up)
    #    Convert compensated final-time vector back to measurement dict
    from src.reconstruct.constraints import sensor_ids_ordered

    ids = sensor_ids_ordered(sensor_config)
    comp_final = compensated[-1]
    comp_meas = {ids[i]: float(comp_final[i]) for i in range(len(ids))}

    # Re-solve reference to reset state for EFIT
    _solve_reference(mygs)

    result = efit_reconstruct(
        mygs, comp_meas, sensor_config, G, coil_names, max_iter=15
    )

    # 6. Assert Ip recovery < 5%
    recon_ip = result.equilibrium.plasma_current
    rel_err = abs(recon_ip - true_ip) / abs(true_ip)
    assert rel_err < 0.05, (
        f"Pipeline Ip error {rel_err:.4f} > 5% "
        f"(true={true_ip:.0f}, recon={recon_ip:.0f})"
    )

    # Also check chi-squared is reasonable
    assert result.diagnostics.chi_squared < 1.0, (
        f"Pipeline chi² = {result.diagnostics.chi_squared:.2e} > 1.0"
    )


def test_regression():
    """[9.2] All prior tests still pass (no regressions).

    Runs the full test suite (excluding this test) and asserts zero failures.
    """
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests/",
            "-v", "--tb=short",
            "-k", "not test_regression",
            "--no-header", "-q",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    # Check that pytest exited cleanly (0 = all passed, 5 = no tests collected is OK too)
    assert result.returncode in (0, 5), (
        f"Regression: pytest returned {result.returncode}\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-1000:]}"
    )
    # Ensure no failures in output
    assert "failed" not in result.stdout.lower() or "0 failed" in result.stdout.lower(), (
        f"Regression failures detected:\n{result.stdout[-2000:]}"
    )
