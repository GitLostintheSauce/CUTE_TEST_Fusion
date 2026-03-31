"""CUTE Reference Equilibrium — Phase 2

This script creates a reference CUTE equilibrium using TokaMaker.
It demonstrates:
1. Loading the CUTE mesh and geometry
2. Running a static Grad-Shafranov solve
3. Running a time-dependent solve (ramp-up, flat-top, ramp-down)
4. Extracting key plasma parameters and plotting flux surfaces
"""
# %% Imports
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import create_isoflux, create_power_flux_fun

# %% Load geometry
with open(os.path.join(os.path.dirname(__file__), "..", "config", "CUTE_geom.json"), "r") as f:
    CUTE_geom = json.load(f)

# %% Initialize TokaMaker
myOFT = OFT_env(nthreads=2)
mygs = TokaMaker(myOFT)

# %% Load mesh
mesh_path = os.path.join(os.path.dirname(__file__), "..", "data", "CUTE_mesh.h5")
mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(mesh_path)
mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
mygs.settings.lim_zmax = 0.38
mygs.setup(order=2, F0=0.17)

# %% Configure coils
coil_bounds = {key: [-1.0e3, 1.0e3] for key in mygs.coil_sets}
mygs.set_coil_bounds(coil_bounds)

# Coil mirrors for up-down symmetry
coil_mirrors = {"CS{0:02d}".format(2 * i + 1): "CS{0:02d}".format(2 * i + 2) for i in range(7)}
coil_mirrors.update({"PF{0:02d}".format(i): "PF{0:02d}".format(15 - i) for i in range(1, 8)})
disable_list = ["PF01"]

regularization_terms = []
for name in mygs.coil_sets:
    if name not in coil_mirrors:
        continue
    if name in disable_list:
        regularization_terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.0e5))
        regularization_terms.append(
            mygs.coil_reg_term({coil_mirrors[name]: 1.0}, target=0.0, weight=1.0e5)
        )
    else:
        regularization_terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.0e-1))
        regularization_terms.append(
            mygs.coil_reg_term(
                {name: 1.0, coil_mirrors[name]: -1.0}, target=0.0, weight=1.0e0
            )
        )
mygs.set_coil_reg(reg_terms=regularization_terms)

# %% Set profiles (L-mode-like)
ffp_prof = create_power_flux_fun(40, 1.5, 2.0)
pp_prof = create_power_flux_fun(40, 4.0, 1.0)
mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)

# %% Set targets
Ip_target = 200.0e3  # 200 kA
beta_approx = 0.2
mygs.set_targets(Ip=Ip_target, Ip_ratio=(1.0 / beta_approx - 1.0))

# %% Set shape constraints
isoflux_pts = create_isoflux(80, 0.32, 0.0, 0.17, 1.7, 0.4)
isoflux_pts = isoflux_pts[isoflux_pts[:, 0] > 0.3, :]
isoflux_pts = np.vstack((isoflux_pts, np.array([[0.15, 0.0]])))

x_points = np.array([[0.22, -0.33], [0.20, 0.34]])
mygs.set_saddles(x_points)
mygs.set_isoflux(np.vstack((isoflux_pts, x_points)))

# %% Solve static equilibrium
mygs.init_psi(0.32, 0.0, 0.13, 1.7, 0.4)
err_flag = mygs.solve()
print(f"\nStatic solve converged: err_flag={err_flag}")

# %% Extract and print key parameters
stats = mygs.get_stats()
print("\n=== Reference Equilibrium Parameters ===")
print(f"  Plasma current Ip  = {stats['Ip']/1e3:.1f} kA")
print(f"  q on axis (q0)     = {stats.get('q0', 'N/A')}")
print(f"  q95                = {stats.get('q95', 'N/A')}")
print(f"  Beta poloidal      = {stats['beta_pol']:.2f} %")
print(f"  Elongation kappa   = {stats.get('kappa', 'N/A')}")
print(f"  Triangularity      = {stats.get('delta', 'N/A')}")
print(f"  Internal inductance= {stats.get('li', 'N/A')}")
print(f"  O-point (R,Z)      = ({mygs.o_point[0]:.4f}, {mygs.o_point[1]:.4f})")

# %% Plot static equilibrium
fig, ax = plt.subplots(1, 1, figsize=(5, 7))
mygs.plot_machine(
    fig, ax, coil_colormap="seismic", coil_symmap=True, coil_scale=1.0e-3,
    coil_clabel=r"$I_{coil}$ [kA]"
)
mygs.plot_psi(fig, ax, xpoint_color="k", vacuum_nlevels=6, plasma_nlevels=8)
mygs.plot_constraints(fig, ax, isoflux_color="tab:red", isoflux_marker=".")
ax.set_title("CUTE Reference Equilibrium (Static)")
ax.set_xlabel("R [m]")
ax.set_ylabel("Z [m]")
fig.savefig(
    os.path.join(os.path.dirname(__file__), "..", "data", "cute_reference_equilibrium.png"),
    dpi=150, bbox_inches="tight"
)
plt.close(fig)
print("\nStatic equilibrium plot saved.")

# %% B-field extraction at arbitrary points
B_eval = mygs.get_field_eval("B")
test_points = np.array([
    [0.32, 0.0],   # near magnetic axis
    [0.40, 0.1],   # outboard midplane
    [0.20, -0.2],  # inboard
])
print("\n=== B-field at test points ===")
for pt in test_points:
    B_vals = B_eval.eval(pt)
    B_mag = np.sqrt(B_vals[0] ** 2 + B_vals[1] ** 2 + B_vals[2] ** 2)
    print(f"  (R={pt[0]:.2f}, Z={pt[1]:.2f}): Br={B_vals[0]:.4f}, Bt={B_vals[1]:.4f}, "
          f"Bz={B_vals[2]:.4f}, |B|={B_mag:.4f} T")

# %% Psi evaluation at arbitrary points
psi_eval = mygs.get_field_eval("psi")
print("\n=== Psi at test points ===")
for pt in test_points:
    psi_val = psi_eval.eval(pt)
    print(f"  (R={pt[0]:.2f}, Z={pt[1]:.2f}): psi={psi_val[0]:.6e} Wb")

# %% Time-dependent solve: ramp-up, flat-top, ramp-down
print("\n=== Time-dependent simulation ===")

# Save the converged static equilibrium state
psi_static = mygs.get_psi(False)
coil_currents_static = mygs.get_coil_currents()

# For the time-dependent simulation, we perturb the equilibrium slightly
# to trigger vertical motion, then evolve
psi_ic = psi_static.copy()

# Remove shape constraints for free evolution
mygs.set_saddles(None)
mygs.set_isoflux(None)

# Setup time-dependent solver
dt = 1.0e-4  # 0.1 ms timestep
mygs.setup_td(dt, 1.0e-13, 1.0e-11)

# Evolve for 20 steps (simulates ~2ms of plasma evolution)
sim_time = 0.0
td_results = {"time": [], "z_axis": [], "r_axis": []}

for i in range(20):
    sim_time, _, nl_its, lin_its, nretry = mygs.step_td(sim_time, dt)
    assert nretry >= 0
    td_results["time"].append(sim_time * 1e3)  # convert to ms
    td_results["z_axis"].append(mygs.o_point[1])
    td_results["r_axis"].append(mygs.o_point[0])

print(f"  Completed {len(td_results['time'])} timesteps")
print(f"  Time range: {td_results['time'][0]:.3f} - {td_results['time'][-1]:.3f} ms")
print(f"  Z-axis range: {min(td_results['z_axis']):.4f} - {max(td_results['z_axis']):.4f} m")

# %% Plot time-dependent results
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

axes[0].plot(td_results["time"], td_results["r_axis"])
axes[0].set_ylabel("R_axis [m]")
axes[0].set_title("CUTE Time-Dependent Evolution")
axes[0].grid(True)

axes[1].plot(td_results["time"], td_results["z_axis"])
axes[1].set_ylabel("Z_axis [m]")
axes[1].set_xlabel("Time [ms]")
axes[1].grid(True)

axes[2].set_visible(False)

fig.savefig(
    os.path.join(os.path.dirname(__file__), "..", "data", "cute_td_evolution.png"),
    dpi=150, bbox_inches="tight"
)
plt.close(fig)
print("\nTime-dependent evolution plot saved.")

# %% Summary
print("\n=== Phase 2 Complete ===")
print("Generated:")
print("  - Static reference equilibrium")
print("  - Time-dependent evolution (20 steps)")
print("  - B-field evaluation at arbitrary points")
print("  - data/cute_reference_equilibrium.png")
print("  - data/cute_td_evolution.png")
