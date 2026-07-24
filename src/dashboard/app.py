"""CUTE tokamak diagnostic dashboard (Plotly Dash application)."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

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
MODELS_DIR = PROJECT_ROOT / "models"
AUTH_TOKEN = os.environ.get("CUTE_DASH_TOKEN", "")

sensor_config = generate_cute_sensors()

# Lazily loaded ML surrogate (model + sensor layout), cached after first use.
_SURROGATE_CACHE: dict = {}


def _load_surrogate():
    """Return (model, layout) for the trained surrogate, or (None, None).

    Loaded lazily and cached so a missing model never breaks the dashboard.
    """
    if "loaded" not in _SURROGATE_CACHE:
        _SURROGATE_CACHE["loaded"] = True
        try:
            from src.ml.dataset import SensorLayout
            from src.ml.mlp import MLPRegressor
            model_path = MODELS_DIR / "surrogate.npz"
            if model_path.exists():
                _SURROGATE_CACHE["model"] = MLPRegressor.load(str(model_path))
                _SURROGATE_CACHE["layout"] = SensorLayout.from_config()
            else:
                _SURROGATE_CACHE["model"] = None
                _SURROGATE_CACHE["layout"] = None
        except Exception:
            _SURROGATE_CACHE["model"] = None
            _SURROGATE_CACHE["layout"] = None
    return _SURROGATE_CACHE.get("model"), _SURROGATE_CACHE.get("layout")


def _load_ensemble():
    """Return (ensemble, calibration scales) for error bars, or (None, None).

    Optional: if the ensemble has not been trained the dashboard still works,
    it just shows point estimates without uncertainty.
    """
    if "ens_loaded" not in _SURROGATE_CACHE:
        _SURROGATE_CACHE["ens_loaded"] = True
        _SURROGATE_CACHE["ensemble"] = None
        _SURROGATE_CACHE["scales"] = None
        try:
            import json

            from src.ml.dataset import PARAM_NAMES
            from src.ml.uncertainty import EnsembleSurrogate
            ens_dir = MODELS_DIR / "ensemble"
            cal_path = MODELS_DIR / "uncertainty_calibration.json"
            if ens_dir.is_dir():
                _SURROGATE_CACHE["ensemble"] = EnsembleSurrogate.load(ens_dir)
                if cal_path.exists():
                    with open(cal_path) as f:
                        cal = json.load(f)
                    factors = cal.get("scale_factors", {})
                    _SURROGATE_CACHE["scales"] = np.array(
                        [factors.get(n, 1.0) for n in PARAM_NAMES]
                    )
        except Exception:
            _SURROGATE_CACHE["ensemble"] = None
            _SURROGATE_CACHE["scales"] = None
    return _SURROGATE_CACHE.get("ensemble"), _SURROGATE_CACHE.get("scales")

# Okabe-Ito colorblind-safe palette
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "gray": "#7f8c99",
    "ink": "#16202b",
}
PLOT_COLORWAY = [
    COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["vermillion"],
    COLORS["sky"], COLORS["purple"], COLORS["yellow"],
]


def _style_figure(fig: go.Figure, title: str, xaxis_title: str = "",
                  yaxis_title: str = "") -> go.Figure:
    """Apply the shared dashboard look: readable fonts, clean grid, safe colors."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color=COLORS["ink"])),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=dict(family="Helvetica Neue, Arial, sans-serif", size=14,
                  color=COLORS["ink"]),
        colorway=PLOT_COLORWAY,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=65, r=25, t=55, b=55),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e6eaf0", zeroline=False,
                     showline=True, linecolor="#c8cfd9", ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor="#e6eaf0", zeroline=False,
                     showline=True, linecolor="#c8cfd9", ticks="outside")
    return fig


def _whatif_slider(param: str, label: str, lo: float, hi: float,
                   step: float, value: float) -> html.Div:
    """One labeled slider row for the what-if explorer."""
    marks = {lo: str(lo), hi: str(hi)}
    return html.Div(className="whatif-slider-row", children=[
        html.Label(label, htmlFor=f"whatif-{param}"),
        dcc.Slider(id=f"whatif-{param}", min=lo, max=hi, step=step,
                   value=value, marks=marks,
                   tooltip={"placement": "bottom", "always_visible": True}),
    ])


def _card(title: str, help_text: str, *children, extra_class: str = "") -> html.Div:
    return html.Div(
        className=f"card {extra_class}".strip(),
        children=[html.H2(title), html.P(help_text, className="card-help"),
                  *children],
    )


def _normalized_flux_grid(boundary_r, boundary_z, n_grid: int = 130):
    """Build an illustrative normalized-flux map from the boundary shape.

    The stored equilibria contain only the boundary contour and scalar
    parameters (no psi map), so this maps each interior point to a
    normalized radius rho relative to the boundary and uses psi_N = rho^2.
    It is a shape-derived visualization of nested flux surfaces, not
    solver output, and is labeled as such in the UI.
    """
    br = np.asarray(boundary_r, dtype=float)
    bz = np.asarray(boundary_z, dtype=float)
    r0, z0 = float(br.mean()), float(bz.mean())

    theta_b = np.mod(np.arctan2(bz - z0, br - r0), 2 * np.pi)
    rad_b = np.hypot(br - r0, bz - z0)
    order = np.argsort(theta_b)
    th_s, rad_s = theta_b[order], rad_b[order]
    # Periodic padding so interpolation wraps cleanly around 2*pi
    th_ext = np.concatenate([th_s - 2 * np.pi, th_s, th_s + 2 * np.pi])
    rad_ext = np.tile(rad_s, 3)

    pad_r = 0.08 * (br.max() - br.min() + 1e-9)
    pad_z = 0.08 * (bz.max() - bz.min() + 1e-9)
    r_axis = np.linspace(br.min() - pad_r, br.max() + pad_r, n_grid)
    z_axis = np.linspace(bz.min() - pad_z, bz.max() + pad_z, n_grid)
    rr, zz = np.meshgrid(r_axis, z_axis)

    th = np.mod(np.arctan2(zz - z0, rr - r0), 2 * np.pi)
    b_rad = np.interp(th, th_ext, rad_ext)
    rho = np.hypot(rr - r0, zz - z0) / np.maximum(b_rad, 1e-12)
    psi_n = np.where(rho <= 1.0, rho ** 2, np.nan)

    return r_axis, z_axis, psi_n, (r0, z0)


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

    # Typed Any: Dash's Graph.config expects a TypedDict, but a plain dict is
    # the documented, working usage; annotating avoids stub false positives.
    graph_config: Any = {
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }

    sidebar = html.Div(className="sidebar", children=[
        _card(
            "Shot Browser",
            "Click a row to load a shot into the viewers. The first shot "
            "loads automatically.",
            html.Div(className="control-row", children=[
                html.Button("Refresh shot list", id="refresh-btn",
                            n_clicks=0),
            ]),
            dash_table.DataTable(  # type: ignore[attr-defined]
                id="shot-table",
                columns=[
                    {"name": "Shot", "id": "shot_number"},
                    {"name": "Ip (A)", "id": "plasma_current",
                     "type": "numeric", "format": {"specifier": ",.0f"}},
                    {"name": "Ch", "id": "n_raw_channels"},
                    {"name": "Eq", "id": "has_equilibrium"},
                ],
                row_selectable="single",
                page_size=12,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#003865", "color": "white",
                    "fontWeight": "600",
                },
                style_cell={
                    "fontFamily": "Helvetica Neue, Arial, sans-serif",
                    "textAlign": "left",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"},
                     "backgroundColor": "#f4f6f9"},
                    {"if": {"state": "selected"},
                     "backgroundColor": "#b9d9eb",
                     "border": "1px solid #003865"},
                ],
            ),
        ),
        _card(
            "About this tool",
            "",
            html.P(
                "This dashboard reviews plasma shots from CUTE, the Columbia "
                "University Tokamak for Education. Magnetic sensors around "
                "the vacuum vessel record each shot; the pipeline filters "
                "those signals and reconstructs the plasma shape and "
                "parameters from them. Each panel shows one stage of that "
                "chain, from raw signals to the reconstructed equilibrium."
            ),
            html.P(
                "Pipeline: HDF5 signal store, bandpass/notch filtering and "
                "drift correction, iterative Grad-Shafranov reconstruction "
                "via the Open Fusion Toolkit (TokaMaker), and this viewer."
            ),
            html.Div(
                "All shots shown here are synthetic test data generated by "
                "the pipeline itself. No experimental CUTE data is included.",
                className="about-note",
            ),
            extra_class="about-card",
        ),
    ])

    main_panel = html.Div(className="main-panel", children=[
        _card(
            "ML Surrogate Reconstruction (live)",
            "A neural network trained to recover plasma parameters from the "
            "130 magnetic diagnostic signals, as a fast alternative to "
            "iterative reconstruction. Click to sample a plasma from the "
            "reduced forward model, add measurement noise, and reconstruct "
            "it. The surrogate runs in microseconds per shot. Where an "
            "ensemble is available, predictions carry calibrated 1-sigma "
            "error bars; errors larger than 1 sigma are highlighted, which "
            "should happen for roughly a third of parameters.",
            html.Div(className="control-row", children=[
                html.Button("Sample and reconstruct a plasma",
                            id="surrogate-btn", n_clicks=0),
                html.Span(id="surrogate-latency", className="latency-pill"),
            ]),
            html.Div(className="surrogate-body", children=[
                html.Div(id="surrogate-table", className="surrogate-table"),
                dcc.Graph(id="surrogate-graph", config=graph_config,
                          style={"height": "360px", "flex": "1"}),
            ]),
            extra_class="surrogate-card",
        ),
        _card(
            "What-if Explorer (live surrogate)",
            "Set a plasma state by hand and watch the reconstruction respond. "
            "The sliders define the true plasma; the reduced forward model "
            "computes the 130 sensor signals that state would produce, and "
            "the surrogate reconstructs the parameters from those signals "
            "alone. Signals here are noise-free, so this isolates the "
            "surrogate's model error from measurement error.",
            html.Div(className="whatif-sliders", children=[
                _whatif_slider("Ip", "Plasma current Ip (kA)",
                               20, 250, 5, 150),
                _whatif_slider("R0", "Major radius R0 (m)",
                               0.28, 0.36, 0.005, 0.32),
                _whatif_slider("Z0", "Vertical position Z0 (m)",
                               -0.06, 0.06, 0.005, 0.0),
                _whatif_slider("a", "Minor radius a (m)",
                               0.05, 0.18, 0.005, 0.10),
            ]),
            html.Div(className="surrogate-body", children=[
                html.Div(id="whatif-table", className="surrogate-table"),
                dcc.Graph(id="whatif-graph", config=graph_config,
                          style={"height": "360px", "flex": "1"}),
            ]),
            dcc.Graph(id="whatif-signals-graph", config=graph_config,
                      style={"height": "260px"}),
            extra_class="surrogate-card",
        ),
        html.Div(className="plot-grid", children=[
            _card(
                "Equilibrium Reconstruction",
                "Reconstructed boundary (separatrix), vessel wall, and "
                "sensor layout in the poloidal plane. The shaded map shows "
                "illustrative nested flux surfaces derived from the "
                "boundary shape; the stored results do not include the "
                "full psi solution.",
                html.Div(className="control-row", children=[
                    html.Label("Time index:", htmlFor="time-slider"),
                ]),
                html.Div(
                    dcc.Slider(id="time-slider", min=0, max=0, step=1,
                               value=0, marks={0: "0"},
                               tooltip={"placement": "bottom",
                                        "always_visible": True}),
                    style={"maxWidth": "460px", "marginBottom": "6px"},
                ),
                dcc.Graph(id="equilibrium-graph", config=graph_config,
                          style={"height": "560px"}),
            ),
            _card(
                "Parameter Timeline",
                "Reconstructed scalar parameters across the shot: plasma "
                "current Ip (left axis, A); safety factor q95, poloidal "
                "beta, and internal inductance li (right axis, "
                "dimensionless).",
                dcc.Graph(id="parameter-graph", config=graph_config,
                          style={"height": "560px"}),
            ),
        ]),

        html.Div(className="plot-grid", children=[
            _card(
                "Signal Viewer",
                "Raw and processed traces for one diagnostic channel "
                "(FL = flux loop, MP = Mirnov probe).",
                html.Div(className="control-row", children=[
                    html.Label("Channel:", htmlFor="channel-dropdown"),
                    dcc.Dropdown(id="channel-dropdown",
                                 placeholder="Select channel",
                                 style={"minWidth": "240px"}),
                ]),
                dcc.Graph(id="signal-graph", config=graph_config),
            ),
            _card(
                "Measured vs. Baseline",
                "Measured signal against a constant mean baseline, with "
                "residual. A full forward-model comparison (synthetic "
                "diagnostics from the reconstructed equilibrium) is the "
                "planned next step and requires a live TokaMaker solve.",
                html.Div(className="control-row", children=[
                    html.Label("Channel:",
                               htmlFor="simexp-channel-dropdown"),
                    dcc.Dropdown(id="simexp-channel-dropdown",
                                 placeholder="Select channel",
                                 style={"minWidth": "240px"}),
                ]),
                dcc.Graph(id="simexp-graph", config=graph_config),
            ),
        ]),

        _card(
            "Export",
            "Use the camera icon in any plot's toolbar to download it as a "
            "high-resolution PNG.",
            html.Button("Download current plot as PNG",
                        id="export-png-btn", n_clicks=0),
            dcc.Download(id="download-data"),
        ),
    ])

    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="auth-store", data={"authenticated": AUTH_TOKEN == ""}),
        dcc.Store(id="current-shot-path", data=None),

        html.Header(className="app-header", children=[
            html.H1("CUTE Diagnostic Dashboard"),
            html.Span("Columbia University Tokamak for Education: magnetic "
                      "equilibrium reconstruction and shot review",
                      className="subtitle"),
            html.Span("SYNTHETIC DEMO DATA", className="badge-demo",
                      title="All shots shown are synthetic test data "
                            "generated by the pipeline; no experimental "
                            "CUTE data is included."),
        ]),

        # Auth gate
        html.Div(id="auth-gate", children=[
            html.Div(className="card auth-card", children=[
                html.H2("Sign in"),
                html.P("Enter the access token to view shot data.",
                       className="card-help"),
                dcc.Input(id="token-input", type="password",
                          placeholder="Access token", debounce=True,
                          n_submit=0),
                html.Button("Log in", id="login-btn", n_clicks=0),
                html.Div(id="auth-msg", className="auth-error", role="alert"),
            ]),
        ], style={"display": "block" if AUTH_TOKEN else "none"}),

        # Main content
        html.Div(id="main-content", className="app-shell",
                 children=[sidebar, main_panel],
                 style={"display": "none" if AUTH_TOKEN else "flex"}),

        html.Footer(
            "CUTE magnetic diagnostic pipeline, Columbia University Tokamak "
            "for Education",
            className="app-footer",
        ),
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
        Input("token-input", "n_submit"),
        State("token-input", "value"),
        State("auth-store", "data"),
        prevent_initial_call=True,
    )
    def handle_login(n_clicks, n_submit, token_value, auth_data):
        if not AUTH_TOKEN:
            return {"authenticated": True}, {"display": "none"}, {"display": "flex"}, ""
        if token_value == AUTH_TOKEN:
            return {"authenticated": True}, {"display": "none"}, {"display": "flex"}, ""
        return ({"authenticated": False}, {"display": "block"},
                {"display": "none"}, "Invalid token. Please try again.")

    @app.callback(
        Output("shot-table", "data"),
        Output("shot-table", "selected_rows"),
        Input("refresh-btn", "n_clicks"),
    )
    def refresh_shot_table(n_clicks):
        rows = get_shot_browser_data()
        # Auto-select the first shot so the page is populated on load.
        return rows, ([0] if rows else [])

    @app.callback(
        Output("current-shot-path", "data"),
        Output("channel-dropdown", "options"),
        Output("channel-dropdown", "value"),
        Output("simexp-channel-dropdown", "options"),
        Output("simexp-channel-dropdown", "value"),
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
        first = channels[0] if channels else None

        n_eq = len(shot.equilibrium) if shot.equilibrium else 0
        max_idx = max(0, n_eq - 1)
        # Sparse marks so long shots stay readable.
        if n_eq > 1:
            step = max(1, n_eq // 10)
            marks = {i: str(i) for i in range(0, n_eq, step)}
            marks[max_idx] = str(max_idx)
        else:
            marks = {0: "0"}

        return shot_path, options, first, options, first, max_idx, marks

    @app.callback(
        Output("signal-graph", "figure"),
        Input("channel-dropdown", "value"),
        Input("current-shot-path", "data"),
    )
    def update_signal_viewer(channel, shot_path):
        return get_signal_viewer_figure(shot_path, channel)

    @app.callback(
        Output("equilibrium-graph", "figure"),
        Input("time-slider", "value"),
        Input("current-shot-path", "data"),
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
        Input("current-shot-path", "data"),
    )
    def update_simexp(channel, shot_path):
        return get_sim_vs_experiment_figure(shot_path, channel)

    @app.callback(
        Output("surrogate-table", "children"),
        Output("surrogate-graph", "figure"),
        Output("surrogate-latency", "children"),
        Input("surrogate-btn", "n_clicks"),
    )
    def run_surrogate(n_clicks):
        return get_surrogate_demo(seed=n_clicks)

    @app.callback(
        Output("whatif-table", "children"),
        Output("whatif-graph", "figure"),
        Output("whatif-signals-graph", "figure"),
        Input("whatif-Ip", "value"),
        Input("whatif-R0", "value"),
        Input("whatif-Z0", "value"),
        Input("whatif-a", "value"),
    )
    def run_whatif(ip_ka, r0, z0, a):
        return get_whatif_demo(ip_ka, r0, z0, a)


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
        return _style_figure(fig, "Select a shot and channel")

    shot = load_shot(shot_path)

    if shot.raw and channel in shot.raw:
        sig = shot.raw[channel]
        fig.add_trace(go.Scatter(
            x=sig.timestamps, y=sig.values, name="Raw",
            line=dict(color=COLORS["gray"], width=1), opacity=0.6,
        ))
    if shot.processed and channel in shot.processed:
        sig = shot.processed[channel]
        fig.add_trace(go.Scatter(
            x=sig.timestamps, y=sig.values, name="Processed",
            line=dict(color=COLORS["blue"], width=2),
        ))

    return _style_figure(fig, f"Signal: {channel}", "Time (s)", "Amplitude")


def get_equilibrium_viewer_figure(
    shot_path: str | None, time_idx: int | None
) -> go.Figure:
    """Return a Plotly figure with flux surface map, boundary, and sensors."""
    fig = go.Figure()
    if not shot_path:
        return _style_figure(fig, "Select a shot")

    shot = load_shot(shot_path)
    if not shot.equilibrium or time_idx is None:
        return _style_figure(fig, "No equilibrium data")

    idx = min(time_idx, len(shot.equilibrium) - 1)
    eq = shot.equilibrium[idx]

    # Illustrative normalized-flux map derived from the boundary shape
    # (the stored equilibria do not include the full psi solution).
    if len(eq.boundary_r) > 2:
        r_axis, z_axis, psi_n, (r0, z0) = _normalized_flux_grid(
            eq.boundary_r, eq.boundary_z
        )
        fig.add_trace(go.Contour(
            x=r_axis, y=z_axis, z=psi_n,
            colorscale="Cividis_r",
            contours=dict(start=0.1, end=0.9, size=0.1,
                          coloring="heatmap",
                          showlines=True),
            line=dict(color="rgba(255,255,255,0.45)", width=1),
            colorbar=dict(
                title=dict(text="psi_N (illustrative)", side="right"),
                thickness=14, len=0.75,
            ),
            name="Flux map",
            hovertemplate="R=%{x:.3f} m<br>Z=%{y:.3f} m<br>"
                          "psi_N=%{z:.2f}<extra>illustrative</extra>",
        ))
        # Geometric center of the boundary (proxy for the magnetic axis;
        # the true axis location is not stored).
        fig.add_trace(go.Scatter(
            x=[r0], y=[z0], mode="markers", name="Axis (geometric)",
            marker=dict(color="white", size=11, symbol="x-thin",
                        line=dict(color=COLORS["ink"], width=2)),
        ))

    # Vessel wall (approximate CUTE VV)
    theta = np.linspace(0, 2 * np.pi, 100)
    vv_r = 0.32 + 0.20 * np.cos(theta)
    vv_z = 0.40 * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=vv_r.tolist(), y=vv_z.tolist(), mode="lines", name="Vessel Wall",
        line=dict(color=COLORS["gray"], width=2, dash="dash"),
        hoverinfo="skip",
    ))

    # Boundary trace (last closed flux surface)
    br = list(eq.boundary_r) + [eq.boundary_r[0]] if len(eq.boundary_r) > 1 else eq.boundary_r
    bz = list(eq.boundary_z) + [eq.boundary_z[0]] if len(eq.boundary_z) > 1 else eq.boundary_z
    fig.add_trace(go.Scatter(
        x=br, y=bz, mode="lines", name="Plasma Boundary (LCFS)",
        line=dict(color=COLORS["vermillion"], width=3.5),
    ))

    # Sensor positions
    fl_r = [s["R"] for s in sensor_config.flux_loops]
    fl_z = [s["Z"] for s in sensor_config.flux_loops]
    fig.add_trace(go.Scatter(
        x=fl_r, y=fl_z, mode="markers", name="Flux Loops",
        marker=dict(color=COLORS["green"], size=8, symbol="circle",
                    line=dict(color="white", width=1)),
    ))

    mp_r = [s["R"] for s in sensor_config.mirnov_probes]
    mp_z = [s["Z"] for s in sensor_config.mirnov_probes]
    fig.add_trace(go.Scatter(
        x=mp_r, y=mp_z, mode="markers", name="Mirnov Probes",
        marker=dict(color=COLORS["orange"], size=8, symbol="diamond",
                    line=dict(color="white", width=1)),
    ))

    _style_figure(
        fig,
        f"Equilibrium at t[{idx}]: Ip = {eq.plasma_current / 1e3:,.1f} kA, "
        f"q95 = {eq.q95:.2f}",
        "R (m)", "Z (m)",
    )
    fig.update_layout(
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        hovermode="closest",
    )
    return fig


def get_parameter_timeline_figure(shot_path: str | None) -> go.Figure:
    """Return a figure with Ip, q95, beta_pol, li vs. time index."""
    fig = go.Figure()
    if not shot_path:
        return _style_figure(fig, "Select a shot")

    shot = load_shot(shot_path)
    if not shot.equilibrium:
        return _style_figure(fig, "No equilibrium data")

    indices = list(range(len(shot.equilibrium)))
    ip = [eq.plasma_current for eq in shot.equilibrium]
    q95 = [eq.q95 for eq in shot.equilibrium]
    beta = [eq.beta_poloidal for eq in shot.equilibrium]
    li = [eq.internal_inductance for eq in shot.equilibrium]

    fig.add_trace(go.Scatter(
        x=indices, y=ip, name="Ip (A)", yaxis="y1",
        line=dict(color=COLORS["blue"], width=2.5),
        hovertemplate="Ip = %{y:,.0f} A<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=indices, y=q95, name="q95", yaxis="y2",
        line=dict(color=COLORS["orange"], width=2),
        hovertemplate="q95 = %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=indices, y=beta, name="beta_pol", yaxis="y2",
        line=dict(color=COLORS["green"], width=2),
        hovertemplate="beta_pol = %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=indices, y=li, name="li", yaxis="y2",
        line=dict(color=COLORS["purple"], width=2),
        hovertemplate="li = %{y:.2f}<extra></extra>",
    ))

    _style_figure(fig, "Parameter Timeline", "Time Index")
    fig.update_layout(
        yaxis=dict(title="Plasma current Ip (A)", side="left"),
        yaxis2=dict(title="q95, beta_pol, li (dimensionless)",
                    overlaying="y", side="right", showgrid=False),
    )
    return fig


def get_sim_vs_experiment_figure(
    shot_path: str | None, channel: str | None
) -> go.Figure:
    """Return a figure with measured, baseline, and residual traces.

    The baseline is currently the signal mean, a placeholder until the
    TokaMaker forward model (src/forward/model.py) is wired in; that
    requires a live solver instance and is not run inside the dashboard.
    """
    fig = go.Figure()
    if not shot_path or not channel:
        return _style_figure(fig, "Select a shot and channel")

    shot = load_shot(shot_path)
    signals = shot.processed or shot.raw
    if not signals or channel not in signals:
        return _style_figure(fig, f"Channel {channel} not found")

    sig = signals[channel]
    measured = sig.values

    baseline = np.full_like(measured, np.nanmean(measured))
    residual = measured - baseline

    fig.add_trace(go.Scatter(
        x=sig.timestamps.tolist(), y=measured.tolist(), name="Measured",
        line=dict(color=COLORS["blue"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=sig.timestamps.tolist(), y=baseline.tolist(),
        name="Baseline (mean)",
        line=dict(color=COLORS["orange"], width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=sig.timestamps.tolist(), y=residual.tolist(), name="Residual",
        line=dict(color=COLORS["gray"], width=1.5),
    ))

    return _style_figure(fig, f"Measured vs. Baseline: {channel}",
                         "Time (s)", "Amplitude")


def _boundary_ellipse(R0, Z0, a, kappa=1.6, n=80):
    """Simple elongated-ellipse boundary for visualizing a plasma shape."""
    t = np.linspace(0, 2 * np.pi, n)
    return R0 + a * np.cos(t), Z0 + kappa * a * np.sin(t)


def get_surrogate_demo(seed: int = 0):
    """Sample a plasma, reconstruct it with the surrogate, and build UI output.

    Returns (table_children, figure, latency_text) for the dashboard callback.
    """
    model, layout = _load_surrogate()
    empty_fig = _style_figure(go.Figure(), "")

    if model is None or layout is None:
        msg = html.Div(
            "No trained surrogate found. Run "
            "'python scripts/train_surrogate.py' to create "
            "models/surrogate.npz, then reload.",
            className="about-note",
        )
        return msg, empty_fig, ""

    from src.ml.dataset import generate_dataset

    # Sample one plasma (with noise) from the reduced forward model.
    X, y_true, _ = generate_dataset(n_samples=1, noise_frac=0.02,
                                    seed=10_000 + int(seed), layout=layout)
    y_true = y_true[0]

    # Prefer the ensemble when available: its spread is the uncertainty of the
    # ensemble *mean*, so the prediction and its error bar must come from the
    # same estimator. Fall back to the single network otherwise.
    ensemble, scales = _load_ensemble()
    y_sigma = None
    if ensemble is not None:
        t0 = time.perf_counter()
        mean, sd = ensemble.predict_with_std(X)
        latency_us = (time.perf_counter() - t0) * 1e6
        y_pred = mean[0]
        y_sigma = sd[0] * (scales if scales is not None else 1.0)
    else:
        t0 = time.perf_counter()
        y_pred = model.predict(X)[0]
        latency_us = (time.perf_counter() - t0) * 1e6

    table = _surrogate_table(y_true, y_pred, y_sigma)

    # Shape comparison plot: true vs. reconstructed boundary.
    fig = go.Figure()
    tr_r, tr_z = _boundary_ellipse(y_true[1], y_true[2], y_true[3])
    pr_r, pr_z = _boundary_ellipse(y_pred[1], y_pred[2], y_pred[3])
    fig.add_trace(go.Scatter(x=tr_r, y=tr_z, mode="lines", name="True plasma",
                             line=dict(color=COLORS["gray"], width=3)))
    fig.add_trace(go.Scatter(x=pr_r, y=pr_z, mode="lines",
                             name="Surrogate reconstruction",
                             line=dict(color=COLORS["vermillion"], width=2,
                                       dash="dash")))
    _style_figure(fig, "True vs. reconstructed plasma shape", "R (m)", "Z (m)")
    fig.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1)

    latency = f"Inference: {latency_us:.0f} us/shot"
    return table, fig, latency


def _surrogate_table(y_true, y_pred, y_sigma=None) -> html.Table:
    """Build the true/predicted/error comparison table shared by the
    surrogate and what-if panels."""
    from src.ml.dataset import PARAM_NAMES, PARAM_UNITS

    def _fmt(name, val):
        return f"{val:,.0f}" if name == "Ip" else f"{val:.4f}"

    cols = [html.Th("Parameter"), html.Th("True"), html.Th("Predicted")]
    if y_sigma is not None:
        cols.append(html.Th("1-sigma"))
    cols.append(html.Th("Abs. error"))
    rows = [html.Tr(cols)]

    for j, (name, unit) in enumerate(zip(PARAM_NAMES, PARAM_UNITS)):
        err = abs(y_true[j] - y_pred[j])
        cells = [
            html.Td(f"{name} ({unit})"),
            html.Td(_fmt(name, y_true[j])),
            html.Td(_fmt(name, y_pred[j])),
        ]
        if y_sigma is not None:
            cells.append(html.Td(f"+/- {_fmt(name, y_sigma[j])}"))
        # Flag when the truth falls outside the stated 1-sigma band; for a
        # calibrated estimate this should happen about a third of the time.
        outside = y_sigma is not None and err > y_sigma[j]
        cells.append(html.Td(
            _fmt(name, err),
            style={"color": COLORS["vermillion"], "fontWeight": "700"}
            if outside else None,
        ))
        rows.append(html.Tr(cells))
    return html.Table(rows)


def get_whatif_demo(ip_ka: float, r0: float, z0: float, a: float):
    """Run the what-if explorer: sliders -> forward model -> surrogate.

    The slider values define the true plasma. The reduced forward model
    computes the noise-free sensor signals for that state, and the surrogate
    reconstructs the parameters from those signals alone.

    Returns (table_children, boundary_figure, signals_figure).
    """
    model, layout = _load_surrogate()
    empty_fig = _style_figure(go.Figure(), "")

    if model is None or layout is None:
        msg = html.Div(
            "No trained surrogate found. Run "
            "'python scripts/train_surrogate.py' to create "
            "models/surrogate.npz, then reload.",
            className="about-note",
        )
        return msg, empty_fig, empty_fig

    from src.ml.dataset import forward_signals

    y_true = np.array([float(ip_ka) * 1e3, float(r0), float(z0), float(a)])
    X = forward_signals(y_true[0], y_true[1], y_true[2], y_true[3],
                        layout)[None, :]

    # Same estimator policy as the surrogate panel: prediction and error bar
    # must come from the same model, ensemble preferred.
    ensemble, scales = _load_ensemble()
    y_sigma = None
    if ensemble is not None:
        mean, sd = ensemble.predict_with_std(X)
        y_pred = mean[0]
        y_sigma = sd[0] * (scales if scales is not None else 1.0)
    else:
        y_pred = model.predict(X)[0]

    table = _surrogate_table(y_true, y_pred, y_sigma)

    # Boundary overlay: the state you set vs. what the surrogate recovered.
    fig = go.Figure()
    tr_r, tr_z = _boundary_ellipse(y_true[1], y_true[2], y_true[3])
    pr_r, pr_z = _boundary_ellipse(y_pred[1], y_pred[2], y_pred[3])
    fig.add_trace(go.Scatter(x=tr_r, y=tr_z, mode="lines", name="Set plasma",
                             line=dict(color=COLORS["gray"], width=3)))
    fig.add_trace(go.Scatter(x=pr_r, y=pr_z, mode="lines",
                             name="Surrogate reconstruction",
                             line=dict(color=COLORS["vermillion"], width=2,
                                       dash="dash")))
    _style_figure(fig, "Set vs. reconstructed plasma shape", "R (m)", "Z (m)")
    fig.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1)

    # Sensor response: what the 130 diagnostics would read for this state.
    n_fl = len(layout.fl_R)
    sig_fig = go.Figure()
    sig_fig.add_trace(go.Scatter(
        x=np.arange(n_fl), y=X[0, :n_fl], mode="lines+markers",
        name="Flux loops (Wb)", marker=dict(size=4),
        line=dict(color=COLORS["blue"], width=1.5)))
    sig_fig.add_trace(go.Scatter(
        x=np.arange(n_fl, layout.n_sensors), y=X[0, n_fl:],
        mode="lines+markers", name="Mirnov probes (T)",
        marker=dict(size=4),
        line=dict(color=COLORS["green"], width=1.5)))
    _style_figure(sig_fig,
                  "Forward-model sensor signals for this state (noise-free)",
                  "Sensor index", "Signal")
    return table, fig, sig_fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CUTE Dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Use 0.0.0.0 to allow access from other machines")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(data_dir=args.data_dir, token=args.token)
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
