"""Phase 5 acceptance tests: Equilibrium reconstruction pipeline."""
import numpy as np
import pytest

oft_available = True
try:
    import OpenFUSIONToolkit  # noqa: F401
except ImportError:
    oft_available = False

pytestmark = pytest.mark.skipif(not oft_available, reason="OFT not installed")


def _generate_synthetic_measurements(mygs, sensor_config, noise_sigma=0.0, rng=None):
    """Generate synthetic measurements from the current equilibrium state."""
    from src.forward.model import full_diagnostic_set

    df = full_diagnostic_set(mygs, sensor_config, time_index=0.0)
    measurements = {}
    for col in df.columns:
        if col == "time":
            continue
        val = float(df[col].iloc[0])
        if noise_sigma > 0 and rng is not None:
            val += rng.normal(0, noise_sigma * abs(val) if abs(val) > 1e-10 else noise_sigma)
        measurements[col] = val
    return measurements


@pytest.fixture(scope="module")
def reference_state(tokamaker_session):
    """Solve reference equilibrium and return (mygs, stats, measurements)."""
    from src.forward.sensors import generate_cute_sensors
    from tests.conftest import _solve_reference

    mygs = tokamaker_session
    _solve_reference(mygs)

    stats = mygs.get_stats()
    sensor_config = generate_cute_sensors()
    measurements = _generate_synthetic_measurements(mygs, sensor_config)

    # Store ground truth coil currents and psi for warm-start tests
    true_coil_currents = dict(mygs.get_coil_currents()[0])

    return mygs, stats, measurements, sensor_config, true_coil_currents


def test_zero_noise_flux_loop_roundtrip(reference_state):
    """[5.1] Zero-noise round-trip: flux loop signals match within 0.1% of range."""
    from src.reconstruct.solver import fit_equilibrium

    mygs, true_stats, measurements, sensor_config, _ = reference_state

    # Re-solve reference to reset state
    from tests.conftest import _solve_reference
    _solve_reference(mygs)

    fit_equilibrium(
        mygs, measurements, sensor_config,
        ip_target=true_stats["Ip"],
        max_iter=5,
        use_jacobian=False,
    )

    # Check flux loop values
    from src.forward.model import flux_loop
    psi_eval = mygs.get_field_eval("psi")

    meas_fl = [measurements[s["id"]] for s in sensor_config.flux_loops]
    psi_range = max(meas_fl) - min(meas_fl) if len(meas_fl) > 1 else 1.0

    for sensor in sensor_config.flux_loops:
        recon_val = flux_loop(psi_eval, (sensor["R"], sensor["Z"]))
        meas_val = measurements[sensor["id"]]
        err = abs(recon_val - meas_val)
        assert err < 0.001 * abs(psi_range), (
            f"{sensor['id']}: error {err:.2e} > 0.1% of range {psi_range:.2e}"
        )


def test_zero_noise_ip_recovery(reference_state):
    """[5.2] Zero-noise round-trip: Ip recovered within 1%."""
    from src.reconstruct.solver import fit_equilibrium

    mygs, true_stats, measurements, sensor_config, _ = reference_state

    from tests.conftest import _solve_reference
    _solve_reference(mygs)

    result = fit_equilibrium(
        mygs, measurements, sensor_config,
        ip_target=true_stats["Ip"],
        max_iter=5,
        use_jacobian=False,
    )

    true_ip = true_stats["Ip"]
    recon_ip = result.equilibrium.plasma_current
    rel_err = abs(recon_ip - true_ip) / abs(true_ip)
    assert rel_err < 0.01, f"Ip error {rel_err:.4f} > 1%: true={true_ip:.0f}, recon={recon_ip:.0f}"


def test_zero_noise_boundary_recovery(reference_state):
    """[5.3] Zero-noise round-trip: boundary points within 0.5 cm."""
    from src.reconstruct.solver import fit_equilibrium

    mygs, true_stats, measurements, sensor_config, _ = reference_state

    # Get ground truth boundary
    from tests.conftest import _solve_reference
    _solve_reference(mygs)

    from src.reconstruct.solver import _get_boundary_rz
    true_r, true_z = _get_boundary_rz(mygs)

    # Reconstruct
    result = fit_equilibrium(
        mygs, measurements, sensor_config,
        ip_target=true_stats["Ip"],
        max_iter=5,
        use_jacobian=False,
    )

    recon_r = np.array(result.equilibrium.boundary_r)
    recon_z = np.array(result.equilibrium.boundary_z)
    true_r = np.array(true_r)
    true_z = np.array(true_z)

    # Compare: for each reconstructed boundary point, find nearest true boundary point
    max_dist = 0.0
    for i in range(len(recon_r)):
        dists = np.sqrt((true_r - recon_r[i])**2 + (true_z - recon_z[i])**2)
        max_dist = max(max_dist, np.min(dists))

    assert max_dist < 0.005, f"Max boundary error {max_dist*100:.2f} cm > 0.5 cm"


def test_noisy_ip_recovery(reference_state):
    """[5.4] Noisy reconstruction: Ip recovered within 2%."""
    from src.reconstruct.solver import fit_equilibrium

    mygs, true_stats, _, sensor_config, _ = reference_state

    from tests.conftest import _solve_reference
    _solve_reference(mygs)

    # Generate noisy measurements (SNR ~20dB means noise ~ 10% of signal)
    rng = np.random.default_rng(42)
    noisy_meas = _generate_synthetic_measurements(mygs, sensor_config, noise_sigma=0.05, rng=rng)

    result = fit_equilibrium(
        mygs, noisy_meas, sensor_config,
        ip_target=true_stats["Ip"],
        max_iter=5,
        use_jacobian=False,
    )

    true_ip = true_stats["Ip"]
    recon_ip = result.equilibrium.plasma_current
    rel_err = abs(recon_ip - true_ip) / abs(true_ip)
    assert rel_err < 0.02, f"Noisy Ip error {rel_err:.4f} > 2%"


def test_noisy_q95_recovery(reference_state):
    """[5.5] Noisy reconstruction: q95 recovered within 10%."""
    from src.reconstruct.solver import fit_equilibrium

    mygs, true_stats, _, sensor_config, _ = reference_state

    from tests.conftest import _solve_reference
    _solve_reference(mygs)

    rng = np.random.default_rng(42)
    noisy_meas = _generate_synthetic_measurements(mygs, sensor_config, noise_sigma=0.05, rng=rng)

    result = fit_equilibrium(
        mygs, noisy_meas, sensor_config,
        ip_target=true_stats["Ip"],
        max_iter=5,
        use_jacobian=False,
    )

    true_q95 = true_stats.get("q_95", 0.0)
    if true_q95 == 0.0:
        pytest.skip("q95 not available for this equilibrium")

    recon_q95 = result.equilibrium.q95
    rel_err = abs(recon_q95 - true_q95) / abs(true_q95)
    assert rel_err < 0.10, f"Noisy q95 error {rel_err:.4f} > 10%"


def test_noisy_boundary_recovery(reference_state):
    """[5.6] Noisy reconstruction: boundary points within 1.0 cm."""
    from src.reconstruct.solver import fit_equilibrium

    mygs, true_stats, _, sensor_config, _ = reference_state

    from tests.conftest import _solve_reference
    _solve_reference(mygs)

    from src.reconstruct.solver import _get_boundary_rz
    true_r, true_z = _get_boundary_rz(mygs)

    rng = np.random.default_rng(42)
    noisy_meas = _generate_synthetic_measurements(mygs, sensor_config, noise_sigma=0.05, rng=rng)

    result = fit_equilibrium(
        mygs, noisy_meas, sensor_config,
        ip_target=true_stats["Ip"],
        max_iter=5,
        use_jacobian=False,
    )

    recon_r = np.array(result.equilibrium.boundary_r)
    recon_z = np.array(result.equilibrium.boundary_z)
    true_r = np.array(true_r)
    true_z = np.array(true_z)

    max_dist = 0.0
    for i in range(len(recon_r)):
        dists = np.sqrt((true_r - recon_r[i])**2 + (true_z - recon_z[i])**2)
        max_dist = max(max_dist, np.min(dists))

    assert max_dist < 0.01, f"Noisy boundary error {max_dist*100:.2f} cm > 1.0 cm"


def test_convergence(reference_state):
    """[5.7] Reconstruction converges in < 50 iterations."""
    from src.reconstruct.solver import fit_equilibrium

    mygs, true_stats, measurements, sensor_config, _ = reference_state

    from tests.conftest import _solve_reference
    _solve_reference(mygs)

    result = fit_equilibrium(
        mygs, measurements, sensor_config,
        ip_target=true_stats["Ip"],
        max_iter=50,
        use_jacobian=False,
    )

    assert result.diagnostics.n_iterations < 50, (
        f"Did not converge in 50 iterations (took {result.diagnostics.n_iterations})"
    )


def test_residual_diagnostics(reference_state):
    """[5.8] Residual diagnostics are fully populated."""
    from src.reconstruct.solver import fit_equilibrium

    mygs, true_stats, measurements, sensor_config, _ = reference_state

    from tests.conftest import _solve_reference
    _solve_reference(mygs)

    result = fit_equilibrium(
        mygs, measurements, sensor_config,
        ip_target=true_stats["Ip"],
        max_iter=5,
        use_jacobian=False,
    )

    diag = result.diagnostics
    assert isinstance(diag.chi_squared, float)
    assert diag.chi_squared >= 0
    assert len(diag.per_sensor_residual) == sensor_config.n_total, (
        f"Got {len(diag.per_sensor_residual)} residuals, expected {sensor_config.n_total}"
    )
    assert isinstance(diag.condition_number, float)
    assert diag.condition_number > 0


def test_timeseries_warmstart(reference_state):
    """[5.9] Warm-started slices converge in fewer iterations than the first."""
    from src.reconstruct.timeseries import reconstruct_timeseries

    mygs, true_stats, measurements, sensor_config, _ = reference_state

    from tests.conftest import _solve_reference
    _solve_reference(mygs)

    # Create 5 identical time slices (same equilibrium)
    measurement_series = [dict(measurements) for _ in range(5)]
    ip_targets = [true_stats["Ip"]] * 5

    results = reconstruct_timeseries(
        mygs, measurement_series, sensor_config,
        ip_targets=ip_targets,
        max_iter=50,
    )

    assert len(results) == 5

    # Warm-started slices (2-5) should converge in <= iterations as slice 1
    first_iters = results[0].diagnostics.n_iterations
    for i in range(1, 5):
        assert results[i].diagnostics.n_iterations <= first_iters, (
            f"Slice {i+1} took {results[i].diagnostics.n_iterations} iters "
            f"> slice 1's {first_iters}"
        )


def test_timeseries_output_structure(reference_state):
    """[5.10] Time-series output has correct structure."""
    from src.reconstruct.timeseries import reconstruct_timeseries

    mygs, true_stats, measurements, sensor_config, _ = reference_state

    from tests.conftest import _solve_reference
    _solve_reference(mygs)

    n_slices = 3
    measurement_series = [dict(measurements) for _ in range(n_slices)]
    ip_targets = [true_stats["Ip"]] * n_slices

    results = reconstruct_timeseries(
        mygs, measurement_series, sensor_config,
        ip_targets=ip_targets,
        max_iter=10,
    )

    assert len(results) == n_slices

    for i, result in enumerate(results):
        eq = result.equilibrium
        assert isinstance(eq.plasma_current, float), f"Slice {i}: Ip not float"
        assert eq.plasma_current > 0, f"Slice {i}: Ip <= 0"
        assert isinstance(eq.boundary_r, list), f"Slice {i}: boundary_r not list"
        assert isinstance(eq.boundary_z, list), f"Slice {i}: boundary_z not list"
        assert len(eq.boundary_r) > 0, f"Slice {i}: empty boundary"
        assert len(eq.boundary_r) == len(eq.boundary_z), f"Slice {i}: boundary R/Z mismatch"
