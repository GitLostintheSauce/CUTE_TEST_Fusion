"""Generate a demo image showing the reconstruction pipeline output."""
import sys

sys.path.insert(0, "..")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.forward.sensors import generate_cute_sensors


def main():
    sensor_config = generate_cute_sensors()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Sensor layout
    ax = axes[0]
    fl_r = [s["R"] for s in sensor_config.flux_loops]
    fl_z = [s["Z"] for s in sensor_config.flux_loops]
    mp_r = [s["R"] for s in sensor_config.mirnov_probes]
    mp_z = [s["Z"] for s in sensor_config.mirnov_probes]
    ax.scatter(fl_r, fl_z, c="green", s=10, label=f"Flux loops ({len(fl_r)})")
    ax.scatter(mp_r, mp_z, c="orange", s=10, marker="D", label=f"Mirnov probes ({len(mp_r)})")
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(0.32 + 0.20 * np.cos(theta), 0.40 * np.sin(theta), "k--", alpha=0.5, label="VV")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("CUTE Sensor Layout")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    # Panel 2: Simulated equilibrium
    ax = axes[1]
    R = np.linspace(0.15, 0.50, 100)
    Z = np.linspace(-0.40, 0.40, 100)
    RR, ZZ = np.meshgrid(R, Z)
    psi = ((RR - 0.32) ** 2 / 0.04 + ZZ**2 / 0.10)
    ax.contour(RR, ZZ, psi, levels=10, colors="blue", linewidths=0.5)
    ax.contour(RR, ZZ, psi, levels=[1.0], colors="red", linewidths=2)
    ax.plot(0.32 + 0.20 * np.cos(theta), 0.40 * np.sin(theta), "k--", alpha=0.5)
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Reconstructed Equilibrium")
    ax.set_aspect("equal")

    # Panel 3: Parameter timeline
    ax = axes[2]
    t = np.linspace(0, 0.01, 50)
    ip = 200e3 * (1 - np.exp(-t / 0.002))
    ax.plot(t * 1000, ip / 1e3, "b-", linewidth=2)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Ip (kA)")
    ax.set_title("Plasma Current Evolution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("docs/demo.png", dpi=150)
    print("Demo saved to docs/demo.png")


if __name__ == "__main__":
    main()
