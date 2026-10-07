"""Figure for scripts/qlknn_poc.py: where CUTE sits against QLKNN's training data.

Ten small panels, one per QLKNN input, each showing CUTE's value across the
plasma radius against the range QLKNN's core training set covered (gray
band). Filled dots are surfaces where all ten inputs are inside a training
box; hollow dots are surfaces where at least one is outside, so the network
is extrapolating there. The bottom panel shows the ion heat flux QLKNN
predicts, with the same filled/hollow encoding.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.coupling.qlknn_inputs import INPUT_NAMES, TRAINING_BOXES  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BAND = "#dcdad3"
SERIES = "#2a78d6"

LABELS = {
    "Ati": "Ati  (ion temp. gradient)", "Ate": "Ate  (electron temp. gradient)",
    "Ane": "Ane  (electron density gradient)", "Ani": "Ani  (ion density gradient)",
    "q": "q  (safety factor)", "smag": "smag  (magnetic shear)",
    "x": "x  (radius, aspect-rescaled)", "Ti_Te": "Ti / Te",
    "LogNuStar": "log10 collisionality", "normni": "ni / ne",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _profile(ax, rho, y, inside) -> None:
    ax.plot(rho, y, color=SERIES, linewidth=2, zorder=3)
    ax.scatter(rho[inside], y[inside], s=22, color=SERIES, zorder=4,
               edgecolors=SERIES, linewidths=1.0)
    ax.scatter(rho[~inside], y[~inside], s=22, facecolors=SURFACE,
               edgecolors=SERIES, linewidths=1.4, zorder=4)


def plot(results: dict, path: Path) -> None:
    p = results["profiles"]
    rho = np.asarray(p["rho"])
    inside = np.asarray(p["inside_training_data"], dtype=bool)
    rho_lim = results["x_rescaling"]["rho_where_x_reaches_training_max"]
    box = TRAINING_BOXES["QLKNN11D (core)"]

    fig = plt.figure(figsize=(13, 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.15], hspace=0.62, wspace=0.32)
    for i, name in enumerate(INPUT_NAMES):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        _style(ax)
        y = np.asarray(p[name])
        lo, hi = box[name]
        ax.axhspan(lo, hi, color=BAND, zorder=1, linewidth=0)
        _profile(ax, rho, y, inside)
        pad = 0.08 * (max(y.max(), hi) - min(y.min(), lo) + 1e-9)
        ymin = min(y.min(), lo) if lo > -1e3 else y.min()
        ax.set_ylim(min(ymin, y.min()) - pad, max(y.max(), hi) + pad)
        ax.axvline(rho_lim, color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)), zorder=2)
        ax.set_title(LABELS[name], fontsize=9, color=TEXT, loc="left")
        ax.set_xlim(0, 1)
        if i // 5 == 1:
            ax.set_xlabel("r / a", fontsize=8, color=TEXT_2)

    ax = fig.add_subplot(gs[2, :])
    _style(ax)
    qi = np.asarray(p["qi_gB"])
    _profile(ax, rho, qi, inside)
    ax.axvline(rho_lim, color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(rho_lim + 0.01, ax.get_ylim()[1] * 0.92,
            f"r/a = {rho_lim:.2f}: beyond here CUTE's fat shape puts x\n"
            "outside QLKNN's training range (aspect ratio)",
            fontsize=8, color=TEXT_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_xlabel("r / a  (normalized radius)", fontsize=9, color=TEXT_2)
    ax.set_title("Ion heat flux QLKNN predicts, gyro-Bohm units "
                 "(hollow = extrapolation, not a trustworthy number)",
                 fontsize=9, color=TEXT, loc="left")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BAND),
        plt.Line2D([], [], color=SERIES, marker="o", markersize=5, linewidth=2),
        plt.Line2D([], [], color=SERIES, marker="o", markersize=5, linewidth=2,
                   markerfacecolor=SURFACE),
    ]
    fig.legend(handles, ["range covered by QLKNN's core training data",
                         "CUTE, all ten inputs inside a training box",
                         "CUTE, at least one input outside (extrapolation)"],
               loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=TEXT_2, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Synthetic CUTE equilibrium + ASSUMED temperature and density "
                 "profiles vs. QLKNN_7_11's training data",
                 fontsize=11, color=TEXT, y=1.03, x=0.5)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
