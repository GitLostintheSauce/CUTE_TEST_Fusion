"""Phase 8c acceptance tests: Sensor placement optimization."""
import numpy as np
import pytest

oft_available = True
try:
    import OpenFUSIONToolkit  # noqa: F401
except ImportError:
    oft_available = False

pytestmark = pytest.mark.skipif(not oft_available, reason="OFT not installed")


@pytest.fixture(scope="module")
def placement_state(greens_session):
    """Green's matrix and sensor IDs for placement tests (from session cache)."""
    from src.reconstruct.constraints import sensor_ids_ordered

    G, coil_names, sensor_config = greens_session
    sensor_ids = sensor_ids_ordered(sensor_config)
    return G, coil_names, sensor_config, sensor_ids


def test_fisher_information_valid(placement_state):
    """[8c.1] Fisher information matrix is symmetric, PSD, correct shape."""
    from src.validation.sensor_placement import fisher_information

    G, coil_names, _, _ = placement_state
    F = fisher_information(G)

    assert F.shape == (len(coil_names), len(coil_names))
    assert not np.any(np.isnan(F)), "F contains NaN"
    assert not np.any(np.isinf(F)), "F contains Inf"

    # Symmetric
    np.testing.assert_allclose(F, F.T, atol=1e-15)

    # Positive semi-definite (all eigenvalues ≥ 0)
    eigvals = np.linalg.eigvalsh(F)
    assert np.all(eigvals >= -1e-12), f"F has negative eigenvalue: {eigvals.min():.2e}"


def test_leave_one_out_ranking(placement_state):
    """[8c.2] Leave-one-out produces non-uniform ranking."""
    from src.validation.sensor_placement import leave_one_out

    G, _, _, sensor_ids = placement_state
    degradation = leave_one_out(G, sensor_ids)

    values = list(degradation.values())
    assert max(values) > 2 * min(values), (
        f"Sensors equally important: max={max(values):.2e}, min={min(values):.2e}"
    )


def test_greedy_selection_monotonic(placement_state):
    """[8c.3] Greedy selection error decreases monotonically."""
    from src.validation.sensor_placement import greedy_forward_selection

    G, _, _, sensor_ids = placement_state
    _, errors = greedy_forward_selection(G, sensor_ids, max_sensors=40)

    # After enough sensors are selected (≥ n_coils), error should decrease
    n_coils = G.shape[1]
    for k in range(n_coils + 1, len(errors)):
        assert errors[k] <= errors[k - 1] * 1.001, (
            f"Error increased at step {k}: {errors[k]:.4e} > {errors[k-1]:.4e}"
        )


def test_optimal_vs_random(placement_state):
    """[8c.4] Optimal sensor set outperforms random set."""
    from src.validation.sensor_placement import (
        greedy_forward_selection,
        reconstruction_error_metric,
    )

    G, _, _, sensor_ids = placement_state
    n_select = 50

    # Optimal selection
    selected_ids, errors = greedy_forward_selection(G, sensor_ids, max_sensors=n_select)
    optimal_error = errors[-1]

    # Random selections
    rng = np.random.default_rng(42)
    random_errors = []
    for _ in range(5):
        random_idx = rng.choice(len(sensor_ids), size=n_select, replace=False)
        G_random = G[random_idx, :]
        random_errors.append(reconstruction_error_metric(G_random))

    mean_random_error = np.mean(random_errors)
    assert optimal_error < mean_random_error, (
        f"Optimal error ({optimal_error:.2e}) >= mean random ({mean_random_error:.2e})"
    )


def test_sensor_type_complementarity(placement_state):
    """[8c.5] Combined sensor types are better than either alone."""
    from src.validation.sensor_placement import sensor_type_analysis

    G, _, sensor_config, _ = placement_state
    results = sensor_type_analysis(G, sensor_config)

    assert results["both"] < results["flux_only"], (
        f"both ({results['both']:.2e}) >= flux_only ({results['flux_only']:.2e})"
    )
    assert results["both"] < results["mirnov_only"], (
        f"both ({results['both']:.2e}) >= mirnov_only ({results['mirnov_only']:.2e})"
    )


def test_minimum_viable_set(placement_state):
    """[8c.6] Minimum viable set ≤ 80 sensors."""
    from src.validation.sensor_placement import find_minimum_viable_set

    G, coil_names, _, sensor_ids = placement_state
    viable_ids, n_viable = find_minimum_viable_set(
        G, sensor_ids, coil_names, max_sensors=80,
    )

    assert n_viable <= 80, f"Minimum viable set has {n_viable} sensors > 80"
    assert n_viable >= G.shape[1], (
        f"Viable set has {n_viable} sensors < n_coils={G.shape[1]}"
    )
