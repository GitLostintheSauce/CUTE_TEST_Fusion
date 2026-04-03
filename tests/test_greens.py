"""Phase 8a acceptance tests: Green's function matrix."""
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
def greens_state(tokamaker_session, greens_session):
    """Green's function matrix for tests (from session cache)."""
    mygs = tokamaker_session
    G, coil_names, sensor_config = greens_session
    return mygs, G, coil_names, sensor_config


def test_greens_matrix_shape(greens_state):
    """[8a.1] Green's function matrix has correct shape, no NaN/Inf."""
    _, G, coil_names, sensor_config = greens_state

    assert G.shape == (sensor_config.n_total, len(coil_names))
    assert G.shape == (130, 28)
    assert not np.any(np.isnan(G)), "G contains NaN"
    assert not np.any(np.isinf(G)), "G contains Inf"


def test_greens_vacuum_consistency(greens_state, vacuum_consistency_data):
    """[8a.2] G @ I_coils matches actual vacuum solve for known currents."""
    _, G, coil_names, _ = greens_state
    y_baseline, y_actual, test_currents = vacuum_consistency_data

    # Subtract baseline to isolate coil contribution
    y_coil_actual = y_actual - y_baseline

    # Predicted from Green's matrix
    I_vec = np.array([test_currents[name] for name in coil_names])
    y_predicted = G @ I_vec

    # They should match within 0.1% (Green's matrix is linear, vacuum is linear)
    rel_err = np.linalg.norm(y_coil_actual - y_predicted) / max(np.linalg.norm(y_coil_actual), 1e-10)
    assert rel_err < 0.001, f"Vacuum consistency error: {rel_err:.4f} > 0.1%"

    # Non-trivial values
    assert np.linalg.norm(y_predicted) > 1e-10, "Vacuum prediction is all zeros"


def test_greens_cache_roundtrip(greens_state):
    """[8a.3] Green's function matrix can be saved and loaded from HDF5."""
    from src.reconstruct.greens import load_greens_matrix, save_greens_matrix

    _, G, coil_names, _ = greens_state

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        tmp_path = f.name

    try:
        save_greens_matrix(tmp_path, G, coil_names)
        G_loaded, names_loaded = load_greens_matrix(tmp_path)

        np.testing.assert_array_equal(G, G_loaded)
        assert coil_names == names_loaded
    finally:
        os.unlink(tmp_path)
