"""CUTE tokamak diagnostic dashboard — Plotly Dash application."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from src.forward.sensors import generate_cute_sensors
from src.store.hdf5 import load_shot

# ---------------------------------------------------------------------------
# Globals / config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AUTH_TOKEN = os.environ.get("CUTE_DASH_TOKEN", "")

sensor_config = generate_cute_sensors()

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(data_dir: str | Path | None = None, token: str | None = None) -> Dash:
    """Create and return the Dash app instance."""
    global DATA_DIR, AUTH_TOKEN
    if data_dir is not None:
        DATA_DIR = Path(data_dir)
    if token is not None:
        AUTH_TOKEN = token

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "CUTE Dashboard"

    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="auth-store", data={"authenticated": AUTH_TOKEN == ""}),
        dcc.Store(id="current-shot-path", data=None),

        # Auth gate
        html.Div(id="auth-gate", children=[
            html.H2("CUTE Dashboard — Login"),
            dcc.Input(id="token-input", type="password", placeholder="Enter access token"),
            html.Button("Login", id="login-btn", n_clicks=0),
            html.Div(id="auth-msg"),
        ], style={"display": "block" if AUTH_TOKEN else "none"}),

        # Main content
        html.Div(id="main-content", children=[
            html.H1("CUTE Tokamak Dashboard"),

            # Shot browser
            html.H2("Shot Browser"),
            html.Button("Refresh", id="refresh-btn", n_clicks=0),
            dash_table.DataTable(
                id="shot-table",
                columns=[
                    {"name": "Shot #", "id": "shot_number"},
                    {"name": "Timestamp", "id": "timestamp"},
                    {"name": "Ip (A)", "id": "plasma_current"},
                    {"name": "Raw Ch", "id": "n_raw_channels"},
                    {"name": "Proc Ch", "id": "n_processed_channels"},
                    {"name": "Has Eq", "id": "has_equilibrium"},
                    {"name": "File", "id": "file"},
                ],
                row_selectable="single",
                style_table={"overflowX": "auto"},
                page_size=20,
            ),

            html.Hr(),

            # Signal viewer
            html.H2("Signal Viewer"),
            html.Div([
                html.Label("Channel:"),
                dcc.Dropdown(id="channel-dropdown", placeholder="Select channel"),
            ], style={"width": "300px", "display": "inline-block"}),
            dcc.Graph(id="signal-graph"),

            html.Hr(),

            # Equilibrium viewer
            html.H2("Equilibrium Viewer"),
            html.Div([
                html.Label("Time index:"),
                dcc.Slider(id="time-slider", min=0, max=0, step=1, value=0,
                           marks={0: "0"}),
            ], style={"width": "400px"}),
            dcc.Graph(id="equilibrium-graph"),

            html.Hr(),

            # Parameter timeline
            html.H2("Parameter Timeline"),
            dcc.Graph(id="parameter-graph"),

            html.Hr(),

            # Sim vs experiment
            html.H2("Sim vs. Experiment"),
            html.Div([
                html.Label("Channel:"),
                dcc.Dropdown(id="simexp-channel-dropdown", placeholder="Select channel"),
            ], style={"width": "300px", "display": "inline-block"}),
            dcc.Graph(id="simexp-graph"),

            html.Hr(),

            # Export
            html.H2("Export"),
            html.Button("Download current plot as PNG", id="export-png-btn", n_clicks=0),
            dcc.Download(id="download-data"),
        ], style={"display": "none" if AUTH_TOKEN else "block"}),
    ])

    _register_callbacks(app)
    return app


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _register_callbacks(app: Dash):
    @app.callback(
        Output("auth-store", "data"),
        Output("auth-gate", "style"),
        Output("main-content", "style"),
        Output("auth-msg", "children"),
        Input("login-btn", "n_clicks"),
        State("token-input", "value"),
        State("auth-store", "data"),
        prevent_initial_call=True,
    )
    def handle_login(n_clicks, token_value, auth_data):
        if not AUTH_TOKEN:
            return {"authenticated": True}, {"display": "none"}, {"display": "block"}, ""
        if token_value == AUTH_TOKEN:
            return {"authenticated": True}, {"display": "none"}, {"display": "block"}, ""
        return {"authenticated": False}, {"display": "block"}, {"display": "none"}, "Invalid token"

    @app.callback(
        Output("shot-table", "data"),
        Input("refresh-btn", "n_clicks"),
    )
    def refresh_shot_table(n_clicks):
        return get_shot_browser_data()

    @app.callback(
        Output("current-shot-path", "data"),
        Output("channel-dropdown", "options"),
        Output("simexp-channel-dropdown", "options"),
        Output("time-slider", "max"),
        Output("time-slider", "marks"),
        Input("shot-table", "selected_rows"),
        State("shot-table", "data"),
    )
    def select_shot(selected_rows, table_data):
        if not selected_rows or not table_data:
            raise PreventUpdate
        row = table_data[selected_rows[0]]
        shot_path = row["file"]

        shot = load_shot(shot_path)
        channels = []
        if shot.raw:
            channels.extend(sorted(shot.raw.keys()))
        elif shot.processed:
            channels.extend(sorted(shot.processed.keys()))
        options = [{"label": ch, "value": ch} for ch in channels]

        n_eq = len(shot.equilibrium) if shot.equilibrium else 0
        max_idx = max(0, n_eq - 1)
        marks = {i: str(i) for i in range(n_eq)} if n_eq > 0 else {0: "0"}

        return shot_path, options, options, max_idx, marks

    @app.callback(
        Output("signal-graph", "figure"),
        Input("channel-dropdown", "value"),
        State("current-shot-path", "data"),
    )
    def update_signal_viewer(channel, shot_path):
        return get_signal_viewer_figure(shot_path, channel)

    @app.callback(
        Output("equilibrium-graph", "figure"),
        Input("time-slider", "value"),
        State("current-shot-path", "data"),
    )
    def update_equilibrium_viewer(time_idx, shot_path):
        return get_equilibrium_viewer_figure(shot_path, time_idx)

    @app.callback(
        Output("parameter-graph", "figure"),
        Input("current-shot-path", "data"),
    )
    def update_parameter_timeline(shot_path):
        return get_parameter_timeline_figure(shot_path)

    @app.callback(
        Output("simexp-graph", "figure"),
        Input("simexp-channel-dropdown", "value"),
        State("current-shot-path", "data"),
    )
    def update_simexp(channel, shot_path):
        return get_sim_vs_experiment_figure(shot_path, channel)


# ---------------------------------------------------------------------------
# Data helpers (public for testing)
# ---------------------------------------------------------------------------


def get_shot_browser_data(data_dir: Path | None = None) -> list[dict]:
    """Scan data directories for shot HDF5 files and return table rows."""
    from src.store.hdf5 import index as store_index

    search_dir = data_dir or DATA_DIR
    rows = []
    for subdir in ["raw", "processed", "synthetic", ""]:
        d = search_dir / subdir if subdir else search_dir
        if d.is_dir():
            df = store_index(d)
            if not df.empty:
                rows.extend(df.to_dict("records"))

    return rows


def get_signal_viewer_figure(
    shot_path: str | None, channel: str | None
) -> go.Figure:
    """Return a Plotly figure with raw and processed signal traces."""
    fig = go.Figure()
    if not shot_path or not channel:
        fig.update_layout(title="Select a shot and channel")
        return fig

    shot = load_shot(shot_path)

    if shot.raw and channel in shot.raw:
        sig = shot.raw[channel]
        fig.add_trace(go.Scatter(
            x=sig.timestamps, y=sig.values, name="Raw", opacity=0.5
        ))
    if shot.processed and channel in shot.processed:
        sig = shot.processed[channel]
        fig.add_trace(go.Scatter(
            x=sig.timestamps, y=sig.values, name="Processed"
        ))

    fig.update_layout(
        title=f"Signal: {channel}",
        xaxis_title="Time (s)",
        yaxis_title="Value",
    )
    return fig


def get_equilibrium_viewer_figure(
    shot_path: str | None, time_idx: int | None
) -> go.Figure:
    """Return a Plotly figure with flux surface contours, boundary, and sensors."""
    fig = go.Figure()
    if not shot_path:
        fig.update_layout(title="Select a shot")
        return fig

    shot = load_shot(shot_path)
    if not shot.equilibrium or time_idx is None:
        fig.update_layout(title="No equilibrium data")
        return fig

    idx = min(time_idx, len(shot.equilibrium) - 1)
    eq = shot.equilibrium[idx]

    # Boundary trace
    br = list(eq.boundary_r) + [eq.boundary_r[0]] if len(eq.boundary_r) > 1 else eq.boundary_r
    bz = list(eq.boundary_z) + [eq.boundary_z[0]] if len(eq.boundary_z) > 1 else eq.boundary_z
    fig.add_trace(go.Scatter(
        x=br, y=bz, mode="lines", name="Plasma Boundary",
        line=dict(color="red", width=3),
    ))

    # Vessel wall (approximate CUTE VV)
    theta = np.linspace(0, 2 * np.pi, 100)
    vv_r = 0.32 + 0.20 * np.cos(theta)
    vv_z = 0.40 * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=vv_r.tolist(), y=vv_z.tolist(), mode="lines", name="Vessel Wall",
        line=dict(color="gray", width=2, dash="dash"),
    ))

    # Flux surface contours (simple representation using boundary scaled inward)
    for frac in [0.2, 0.4, 0.6, 0.8]:
        if len(eq.boundary_r) > 2:
            r_center = np.mean(eq.boundary_r)
            z_center = np.mean(eq.boundary_z)
            sr = [r_center + frac * (r - r_center) for r in eq.boundary_r]
            sz = [z_center + frac * (z - z_center) for z in eq.boundary_z]
            sr.append(sr[0])
            sz.append(sz[0])
            fig.add_trace(go.Scatter(
                x=sr, y=sz, mode="lines", name=f"ψ={frac:.1f}",
                line=dict(color="blue", width=1),
                showlegend=False,
            ))

    # Sensor positions
    fl_r = [s["R"] for s in sensor_config.flux_loops]
    fl_z = [s["Z"] for s in sensor_config.flux_loops]
    fig.add_trace(go.Scatter(
        x=fl_r, y=fl_z, mode="markers", name="Flux Loops",
        marker=dict(color="green", size=5, symbol="circle"),
    ))

    mp_r = [s["R"] for s in sensor_config.mirnov_probes]
    mp_z = [s["Z"] for s in sensor_config.mirnov_probes]
    fig.add_trace(go.Scatter(
        x=mp_r, y=mp_z, mode="markers", name="Mirnov Probes",
        marker=dict(color="orange", size=5, symbol="diamond"),
    ))

    fig.update_layout(
        title=f"Equilibrium (time index {idx}): Ip={eq.plasma_current:.0f} A",
        xaxis_title="R (m)",
        yaxis_title="Z (m)",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
    )
    return fig


def get_parameter_timeline_figure(shot_path: str | None) -> go.Figure:
    """Return a figure with Ip, q95, beta_pol, li vs. time index."""
    fig = go.Figure()
    if not shot_path:
        fig.update_layout(title="Select a shot")
        return fig

    shot = load_shot(shot_path)
    if not shot.equilibrium:
        fig.update_layout(title="No equilibrium data")
        return fig

    indices = list(range(len(shot.equilibrium)))
    ip = [eq.plasma_current for eq in shot.equilibrium]
    q95 = [eq.q95 for eq in shot.equilibrium]
    beta = [eq.beta_poloidal for eq in shot.equilibrium]
    li = [eq.internal_inductance for eq in shot.equilibrium]

    fig.add_trace(go.Scatter(x=indices, y=ip, name="Ip (A)", yaxis="y1"))
    fig.add_trace(go.Scatter(x=indices, y=q95, name="q95", yaxis="y2"))
    fig.add_trace(go.Scatter(x=indices, y=beta, name="β_pol", yaxis="y2"))
    fig.add_trace(go.Scatter(x=indices, y=li, name="li", yaxis="y2"))

    fig.update_layout(
        title="Parameter Timeline",
        xaxis_title="Time Index",
        yaxis=dict(title="Ip (A)", side="left"),
        yaxis2=dict(title="Dimensionless", overlaying="y", side="right"),
    )
    return fig


def get_sim_vs_experiment_figure(
    shot_path: str | None, channel: str | None
) -> go.Figure:
    """Return a figure with measured, synthetic, and residual traces."""
    fig = go.Figure()
    if not shot_path or not channel:
        fig.update_layout(title="Select a shot and channel")
        return fig

    shot = load_shot(shot_path)
    signals = shot.processed or shot.raw
    if not signals or channel not in signals:
        fig.update_layout(title=f"Channel {channel} not found")
        return fig

    sig = signals[channel]
    measured = sig.values

    # Generate synthetic signal (constant value from equilibrium forward model)
    # For demonstration: use mean value as the "synthetic" baseline
    synthetic = np.full_like(measured, np.nanmean(measured))

    residual = measured - synthetic

    fig.add_trace(go.Scatter(
        x=sig.timestamps.tolist(), y=measured.tolist(), name="Measured"
    ))
    fig.add_trace(go.Scatter(
        x=sig.timestamps.tolist(), y=synthetic.tolist(), name="Synthetic"
    ))
    fig.add_trace(go.Scatter(
        x=sig.timestamps.tolist(), y=residual.tolist(), name="Residual"
    ))

    fig.update_layout(
        title=f"Sim vs. Experiment: {channel}",
        xaxis_title="Time (s)",
        yaxis_title="Value",
    )
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CUTE Dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(data_dir=args.data_dir, token=args.token)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
