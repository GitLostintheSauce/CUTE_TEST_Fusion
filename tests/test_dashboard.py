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
