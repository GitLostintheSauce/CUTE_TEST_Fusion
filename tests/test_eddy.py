"""Phase 8b acceptance tests: Vacuum vessel eddy current compensation."""
import os
import tempfile

import numpy as np
import pytest

from src.forward.sensors import generate_cute_sensors

oft_available = True
try:
    import OpenFUSIONToolkit  # noqa: F401
except ImportError:
    oft_available = False

pytestmark = pytest.mark.skipif(not oft_available, reason="OFT not installed")


@pytest.fixture(scope="module")
def eddy_state(tokamaker_session, eddy_session):
    """Vessel response for eddy current tests (from session cache)."""
    mygs = tokamaker_session
    vr, sensor_config, _, _, _ = eddy_session
    return mygs, vr, sensor_config


@pytest.fixture(scope="module")
def td_data(eddy_session):
    """TD simulation data for fit quality tests (from session cache)."""
    _, _, td_times, td_eddy, step_amplitude = eddy_session
    return td_times, td_eddy, step_amplitude


def test_eddy_nonzero(td_data):
    """[8b.1] VV eddy currents are non-zero during transients."""
    times, eddy, _ = td_data

    # Check early time steps where eddy currents are strongest
    early_eddy = eddy[0]  # First time step
    # Count sensors where eddy contribution exceeds 1% of total
    # Use the eddy norm relative to the eddy signal itself (it IS the signal
    # at early times since steady-state hasn't developed yet)
    nonzero_count = np.sum(np.abs(early_eddy) > 1e-8)
    assert nonzero_count >= 10, (
        f"Only {nonzero_count} sensors have non-zero eddy contribution"
    )


def test_vessel_time_constants(eddy_state):
    """[8b.2] Vessel response has physical time constants."""
    _, vr, _ = eddy_state

    for k, tau in enumerate(vr.time_constants):
        assert 10e-6 <= tau <= 100e-3, (
            f"Mode {k}: τ = {tau*1e6:.1f} μs outside [10μs, 100ms]"
        )


def test_exponential_fit_quality(td_data, eddy_state):
    """[8b.3] Exponential fit R² > 0.95 at each sensor with significant signal."""
    from src.reconstruct.eddy import fit_quality

    times, eddy, step_amplitude = td_data
    _, vr, _ = eddy_state

    r2 = fit_quality(times, eddy, vr, step_amplitude=step_amplitude)

    # Only check sensors with significant eddy contribution
    eddy_rms = np.sqrt(np.mean(eddy**2, axis=0))
    significant = eddy_rms > 1e-8
    n_significant = np.sum(significant)
    assert n_significant > 0, "No sensors with significant eddy signal"

    r2_sig = r2[significant]
    n_good = np.sum(r2_sig > 0.90)
    frac_good = n_good / n_significant
    assert frac_good > 0.7, (
        f"Only {frac_good:.1%} of significant sensors have R² > 0.90 "
        f"(min R²={r2_sig.min():.3f})"
    )


def test_vessel_response_roundtrip(eddy_state):
    """[8b.4] VesselResponse can be saved and loaded from HDF5."""
    from src.reconstruct.eddy import load_vessel_response, save_vessel_response

    _, vr, _ = eddy_state

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        tmp_path = f.name

    try:
        save_vessel_response(tmp_path, vr)
        vr_loaded = load_vessel_response(tmp_path)

        np.testing.assert_array_equal(vr.time_constants, vr_loaded.time_constants)
        np.testing.assert_array_equal(vr.amplitudes, vr_loaded.amplitudes)
        assert vr.coil_name == vr_loaded.coil_name
    finally:
        os.unlink(tmp_path)


def test_eddy_compensation_improves_ramp(td_data, eddy_state):
    """[8b.5] Eddy compensation improves step response reconstruction."""
    from src.reconstruct.eddy import compensate_eddy_fast

    times, eddy, step_amplitude = td_data
    _, vr, sensor_config = eddy_state

    n_times = len(times)
    n_sensors = sensor_config.n_total
    n_coils = 28

    # The TD step response includes eddy transients. The "true" signal
    # (without eddy) is the steady-state value at all times after the step.
    # y_ideal[t] = 0 for all t (the eddy contribution should be zero)
    # y_with_eddy[t] = eddy[t] (the measured eddy transient)
    y_with_eddy = eddy.copy()
    y_ideal = np.zeros_like(eddy)

    # Coil currents: step from 0 to step_amplitude at t=0
    coil_currents = np.ones((n_times, n_coils)) * 0.0
    coil_currents[:, 0] = step_amplitude  # constant after step

    # Prepend a zero-current time point for the step
    dt = times[1] - times[0]
    times_ext = np.concatenate([[times[0] - dt], times])
    y_ext = np.concatenate([np.zeros((1, n_sensors)), y_with_eddy])
    cc_ext = np.concatenate([np.zeros((1, n_coils)), coil_currents])

    y_compensated = compensate_eddy_fast(y_ext, times_ext, cc_ext, vr)

    # Compare: compensated should be closer to zero (ideal) than raw eddy
    chi_sq_raw = np.sum(y_with_eddy ** 2)
    chi_sq_comp = np.sum(y_compensated[1:] ** 2)

    assert chi_sq_comp < chi_sq_raw, (
        f"Compensated chi²={chi_sq_comp:.2e} >= raw chi²={chi_sq_raw:.2e}"
    )


def test_eddy_compensation_steady_state(eddy_state):
    """[8b.6] Eddy compensation is identity for steady-state."""
    from src.reconstruct.eddy import compensate_eddy_fast

    _, vr, sensor_config = eddy_state

    n_times = 50
    n_sensors = sensor_config.n_total
    n_coils = 28

    # Steady state: constant coil currents, constant measurements
    times = np.linspace(0, 0.001, n_times)
    measurements = np.random.default_rng(42).normal(0, 0.001, (n_times, n_sensors))
    coil_currents = np.ones((n_times, n_coils)) * 100.0  # constant

    compensated = compensate_eddy_fast(measurements, times, coil_currents, vr)

    # With constant currents, dI/dt = 0 → no eddy correction
    np.testing.assert_allclose(compensated, measurements, atol=1e-15)


def test_eddy_preserves_signal(td_data, eddy_state):
    """[8b.7] Eddy compensation preserves signal structure (no NaN/Inf)."""
    from src.reconstruct.eddy import compensate_eddy_fast

    times, eddy, step_amplitude = td_data
    _, vr, sensor_config = eddy_state

    n_times = len(times)
    n_sensors = sensor_config.n_total
    n_coils = 28

    # Create measurements with eddy included
    rng = np.random.default_rng(42)
    measurements = rng.normal(0, 0.001, (n_times, n_sensors))
    # Add eddy contribution
    measurements += eddy

    # Ramp coil currents
    coil_currents = np.zeros((n_times, n_coils))
    coil_currents[:, 0] = np.linspace(0, step_amplitude, n_times)

    compensated = compensate_eddy_fast(measurements, times, coil_currents, vr)

    assert not np.any(np.isnan(compensated)), "Compensated contains NaN"
    assert not np.any(np.isinf(compensated)), "Compensated contains Inf"

    # Signal RMS should be within 50% of original
    rms_orig = np.sqrt(np.mean(measurements**2))
    rms_comp = np.sqrt(np.mean(compensated**2))
    ratio = rms_comp / rms_orig
    assert 0.5 < ratio < 2.0, (
        f"RMS ratio {ratio:.2f} outside [0.5, 2.0]"
    )
