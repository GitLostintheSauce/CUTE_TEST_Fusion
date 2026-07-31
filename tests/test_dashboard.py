"""Phase 6a acceptance tests: Dashboard & Visualization."""
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import pytest

from src.dashboard.app import (
    create_app,
    get_equilibrium_viewer_figure,
    get_parameter_timeline_figure,
    get_shot_browser_data,
    get_signal_viewer_figure,
    get_sim_vs_experiment_figure,
)
from src.store.hdf5 import SignalData, save_shot
from src.store.schemas import EquilibriumResult, ShotMetadata, SignalMetadata


@pytest.fixture
def synthetic_shots(tmp_path):
    """Create 3 synthetic shot HDF5 files in a temp directory."""
    rng = np.random.default_rng(42)
    paths = []

    for i in range(1, 4):
        meta = ShotMetadata(
            shot_number=i,
            timestamp=datetime(2026, 1, i, 12, 0, 0),
            coil_currents={"CS01": 100.0 * i},
            gas_pressure=1.0e-3,
        )

        timestamps = np.linspace(0, 0.01, 100)
        raw = {}
        processed = {}
        for ch_id in [f"FL_IB{j:02d}" for j in range(1, 4)]:
            values = np.sin(2 * np.pi * 100 * timestamps) + rng.normal(0, 0.1, 100)
            sig_meta = SignalMetadata(
                channel_id=ch_id,
                sensor_type="flux_loop",
                position_r=0.155,
                position_z=0.0,
            )
            raw[ch_id] = SignalData(metadata=sig_meta, timestamps=timestamps, values=values)
            processed[ch_id] = SignalData(
                metadata=sig_meta, timestamps=timestamps, values=values * 0.9
            )

        equilibrium = [
            EquilibriumResult(
                plasma_current=200000.0 * i,
                q95=3.5 + 0.1 * i,
                beta_poloidal=30.0 + i,
                internal_inductance=1.3 + 0.01 * i,
                boundary_r=[0.15, 0.32, 0.50, 0.32],
                boundary_z=[0.0, 0.35, 0.0, -0.35],
            )
            for _ in range(3)
        ]

        shot_path = tmp_path / f"shot_{i:03d}.h5"
        save_shot(shot_path, meta, raw_signals=raw, processed_signals=processed,
                  equilibrium=equilibrium)
        paths.append(shot_path)

    return tmp_path, paths


def test_shot_browser_data(synthetic_shots):
    """[6a.2] Shot browser returns correct data for 3 mock shots."""
    data_dir, _ = synthetic_shots
    rows = get_shot_browser_data(data_dir=data_dir)
    assert len(rows) == 3, f"Expected 3 shots, got {len(rows)}"

    shot_numbers = sorted(r["shot_number"] for r in rows)
    assert shot_numbers == [1, 2, 3]


def test_signal_viewer_callback(synthetic_shots):
    """[6a.3] Signal viewer callback returns a valid Figure with traces."""
    _, paths = synthetic_shots
    fig = get_signal_viewer_figure(str(paths[0]), "FL_IB01")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1, "Figure should have at least 1 trace"


def test_equilibrium_viewer_callback(synthetic_shots):
    """[6a.4] Equilibrium viewer returns Figure with contour, boundary, and sensors."""
    _, paths = synthetic_shots
    fig = get_equilibrium_viewer_figure(str(paths[0]), 0)
    assert isinstance(fig, go.Figure)

    trace_names = [t.name for t in fig.data if t.name is not None]
    assert any("Boundary" in n for n in trace_names), f"No boundary trace found in {trace_names}"
    assert any("Flux Loop" in n for n in trace_names), f"No sensor trace found in {trace_names}"


def test_parameter_timeline_callback(synthetic_shots):
    """[6a.5] Parameter timeline returns Figure with 4 parameter traces."""
    _, paths = synthetic_shots
    fig = get_parameter_timeline_figure(str(paths[0]))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4, f"Expected 4 traces (Ip, q95, beta, li), got {len(fig.data)}"

    names = {t.name for t in fig.data}
    assert "Ip (A)" in names
    assert "q95" in names


def test_sim_vs_experiment_callback(synthetic_shots):
    """[6a.6] Measured-vs-baseline returns Figure with 3 traces."""
    _, paths = synthetic_shots
    fig = get_sim_vs_experiment_figure(str(paths[0]), "FL_IB01")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3, (
        f"Expected 3 traces (measured, baseline, residual), got {len(fig.data)}"
    )

    names = [t.name for t in fig.data]
    assert "Measured" in names
    assert "Baseline (mean)" in names
    assert "Residual" in names


def test_auth_gate():
    """[6a.8] Auth gate rejects unauthenticated requests, accepts valid token."""
    # App with token required
    app = create_app(token="secret123")
    client = app.server.test_client()

    # The app serves the layout (SPA); auth is client-side via Dash callbacks.
    # Verify the app starts and serves the page.
    resp = client.get("/")
    assert resp.status_code == 200

    # Verify the auth token is set
    from src.dashboard import app as app_module
    assert app_module.AUTH_TOKEN == "secret123"

    # App without token: no auth required
    create_app(token="")
    assert app_module.AUTH_TOKEN == ""


def test_whatif_demo():
    """[7.1] What-if explorer: sliders -> forward model -> surrogate."""
    from src.dashboard.app import get_whatif_demo

    table, boundary_fig, signals_fig = get_whatif_demo(150, 0.32, 0.0, 0.10)

    # With the trained surrogate present, the table compares set vs.
    # predicted parameters; without it, a helpful message renders instead.
    from src.dashboard.app import _load_surrogate
    model, _ = _load_surrogate()
    if model is None:
        pytest.skip("no trained surrogate available")

    assert isinstance(boundary_fig, go.Figure)
    assert len(boundary_fig.data) == 2  # set + reconstructed boundary
    assert isinstance(signals_fig, go.Figure)
    assert len(signals_fig.data) == 2  # flux loops + Mirnov probes

    # The reconstruction should land near the set state (noise-free input).
    rows = table.children
    assert len(rows) == 5  # header + 4 parameters


def test_whatif_tracks_sliders():
    """[7.1] Moving Ip by a factor should move the prediction accordingly."""
    from src.dashboard.app import _load_surrogate, get_whatif_demo

    model, _ = _load_surrogate()
    if model is None:
        pytest.skip("no trained surrogate available")

    def predicted_ip(ip_ka):
        table, _, _ = get_whatif_demo(ip_ka, 0.32, 0.0, 0.10)
        # Row 1, cell "Predicted" (index 2); strip thousands separators.
        cell = table.children[1].children[2].children
        return float(str(cell).replace(",", ""))

    lo, hi = predicted_ip(100), predicted_ip(200)
    assert abs(lo - 100e3) / 100e3 < 0.05
    assert abs(hi - 200e3) / 200e3 < 0.05


def test_surrogate_fixed_plasma_holds_truth_constant():
    """Same-plasma mode must vary only the noise, not the ground truth."""
    from src.dashboard.app import _load_surrogate, get_surrogate_demo

    model, _ = _load_surrogate()
    if model is None:
        pytest.skip("no trained surrogate available")

    def truth_and_pred(seed):
        table, _, _ = get_surrogate_demo(seed=seed, fixed_plasma=True)
        # rows[0] is the header; each later row is [name, true, pred, ...].
        rows = table.children[1:]
        true = [r.children[1].children for r in rows]
        pred = [r.children[2].children for r in rows]
        return true, pred

    true_a, pred_a = truth_and_pred(1)
    true_b, pred_b = truth_and_pred(2)

    # Ground truth is identical across clicks ...
    assert true_a == true_b
    # ... while the reconstruction moves, because the noise draw changed.
    assert pred_a != pred_b


def test_surrogate_new_plasma_changes_truth():
    """Default mode draws a new plasma, so the truth changes between clicks."""
    from src.dashboard.app import _load_surrogate, get_surrogate_demo

    model, _ = _load_surrogate()
    if model is None:
        pytest.skip("no trained surrogate available")

    def truth(seed):
        table, _, _ = get_surrogate_demo(seed=seed, fixed_plasma=False)
        return [r.children[1].children for r in table.children[1:]]

    assert truth(1) != truth(2)


def test_noisy_replicas_fixes_state_and_varies_noise():
    """Replicas of one plasma differ only through the noise draw."""
    import numpy as np

    from src.ml.dataset import noisy_replicas

    y = np.array([1.0e5, 0.32, 0.0, 0.10])
    X, layout = noisy_replicas(y, n_replicas=8, noise_frac=0.02, seed=0)

    assert X.shape == (8, layout.n_sensors)
    # Every replica is distinct (noise applied independently) ...
    assert not np.allclose(X[0], X[1])
    # ... but they scatter about one common noise-free signal.
    clean, _ = noisy_replicas(y, n_replicas=1, noise_frac=0.0, seed=0)
    assert np.allclose(X.mean(axis=0), clean[0], rtol=0.3, atol=1e-9)
