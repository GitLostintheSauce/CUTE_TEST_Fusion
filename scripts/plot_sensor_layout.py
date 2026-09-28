"""Draw CUTE's magnetic diagnostics on the machine cross section.

Output:
    docs/cute_sensors.png

The figure is the poloidal slice (R, Z) used everywhere in this project: the
vacuum vessel wall, the poloidal field coils, the flux loops, and the magnetic
probes with an arrow along the field component each one measures. It is meant
for reading alongside Part 9 of docs/primer.md.

Run:
    python scripts/plot_sensor_layout.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.forward.sensors import generate_cute_sensors  # noqa: E402

BLUE, ORANGE, GREEN, PURPLE, GRAY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#7f8c99"


def main() -> None:
    geom = json.loads((PROJECT_ROOT / "config" / "CUTE_geom.json").read_text())
    config = generate_cute_sensors()

    fig, ax = plt.subplots(figsize=(6.5, 9))

    # Vacuum vessel: the wall is the band between the inner and outer contours.
    for key in ("inner_contour", "outer_contour"):
        c = np.array(geom["vv"][key] + geom["vv"][key][:1])
        ax.plot(c[:, 0], c[:, 1], color="black", lw=1.0)

    # Coils, drawn as their rectangular winding packs.
    for i, coil in enumerate(geom["coils"].values()):
        ax.add_patch(Rectangle(
            (coil["rc"] - coil["w"] / 2, coil["zc"] - coil["h"] / 2), coil["w"], coil["h"],
            facecolor="#d9dde2", edgecolor=GRAY, lw=0.6,
            label="coils (CS: central solenoid, PF: poloidal field)" if i == 0 else None,
        ))

    fs = [s for s in config.flux_loops if s["id"].startswith("FS")]
    fc = [s for s in config.flux_loops if s["id"].startswith("FC")]
    ax.scatter([s["R"] for s in fs], [s["Z"] for s in fs], s=18, color=BLUE, zorder=3,
               label=f"flux loops on the vessel (FS, {len(fs)})")
    ax.scatter([s["R"] for s in fc], [s["Z"] for s in fc], s=18, color=GREEN, zorder=3,
               label=f"flux loops on the center column (FC, {len(fc)})")

    # Probes: arrow along (cos angle, sin angle), the field component measured.
    # A probe and its @180 partner share a position, so their arrows start at
    # the same point and differ only in direction.
    arrow = 0.045
    for kind, color in (("0 deg", ORANGE), ("180 deg", PURPLE)):
        group = [p for p in config.mirnov_probes if p["id"].endswith("_180") == (kind == "180 deg")]
        ax.quiver([p["R"] for p in group], [p["Z"] for p in group],
                  [arrow * np.cos(p["angle"]) for p in group],
                  [arrow * np.sin(p["angle"]) for p in group],
                  angles="xy", scale_units="xy", scale=1, width=0.005, color=color, zorder=4,
                  label=f"magnetic probes at {kind} toroidally ({len(group)})")

    for s in config.flux_loops + config.mirnov_probes:
        if s["id"] in ("FS01", "FS12", "FS25", "FC01", "FC14", "S01", "S07", "S13"):
            ax.annotate(s["id"].replace("_180", ""), (s["R"], s["Z"]), fontsize=7,
                        xytext=(4, 3), textcoords="offset points", color="#333333")

    # Nominal plasma for scale: R0 = 0.32 m, a = 0.17 m, elongation 1.7.
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.32 + 0.17 * np.cos(t + 0.4 * np.sin(t)), 1.7 * 0.17 * np.sin(t),
            color=GRAY, ls="--", lw=1, label="a nominal plasma boundary (for scale)")

    ax.set_aspect("equal")
    ax.set_xlim(0.0, 0.72)
    ax.set_ylim(-0.95, 0.95)
    ax.set_xlabel("R (m): distance from the machine's central axis")
    ax.set_ylabel("Z (m): height")
    ax.set_title("CUTE magnetic diagnostics on the poloidal cross section\n"
                 f"{config.n_flux_loops} flux loops, {config.n_mirnov_probes} probe channels "
                 f"({config.n_total} in total)", fontsize=11)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.95)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out = PROJECT_ROOT / "docs" / "cute_sensors.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
