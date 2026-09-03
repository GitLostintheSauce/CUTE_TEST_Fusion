# TokaMaker API Notes

Notes from Phase 2 exploration of OFT v1.0.0-beta7 TokaMaker.

## Initialization

Only ONE TokaMaker instance per Python kernel. Create the OFT environment first:

```python
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh, save_gs_mesh, gs_Domain
from OpenFUSIONToolkit.TokaMaker.util import create_isoflux, create_power_flux_fun

myOFT = OFT_env(nthreads=2)
mygs = TokaMaker(myOFT)
```

## Mesh Setup

### Building from scratch (gs_Domain)
```python
gs_mesh = gs_Domain()
gs_mesh.define_region('air', vac_resolution, 'boundary')
gs_mesh.define_region('plasma', plasma_resolution, 'plasma')
gs_mesh.define_region('vv', vv_resolution, 'conductor', eta=1.26e-5)
for key, coil in geom['coils'].items():
    gs_mesh.define_region(key, coil_resolution, 'coil', nTurns=coil['nturns'])

# Add shapes
gs_mesh.add_annulus(inner_contour, 'plasma', outer_contour, 'vv', parent_name='air')
for key, coil in geom['coils'].items():
    gs_mesh.add_rectangle(coil['rc'], coil['zc'], coil['w'], coil['h'], key, parent_name='air')

mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()
coil_dict = gs_mesh.get_coils()
cond_dict = gs_mesh.get_conductors()
save_gs_mesh(mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict, 'mesh.h5')
```

### Loading a pre-built mesh
```python
mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh('CUTE_mesh.h5')
mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
mygs.settings.lim_zmax = 0.38  # Prevent limiting in divertor areas
mygs.setup(order=2, F0=0.17)   # order=2-4, F0 = B0*R0
```

Region types: `plasma`, `vacuum`, `boundary`, `conductor` (with `eta`), `coil` (with `nTurns`)

## Coil Configuration

```python
# Current bounds (per turn)
coil_bounds = {key: [-1.E3, 1.E3] for key in mygs.coil_sets}
mygs.set_coil_bounds(coil_bounds)

# Regularization for up-down symmetry + amplitude damping
terms = []
terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.E-1))
terms.append(mygs.coil_reg_term({name: 1.0, mirror: -1.0}, target=0.0, weight=1.0))
mygs.set_coil_reg(reg_terms=terms)

# Set coil currents directly
mygs.set_coil_currents({'CS01': 500.0, ...})

# Read coil currents
currents = mygs.get_coil_currents()  # dict
```

## Profiles

Use `create_power_flux_fun(npoints, alpha, gamma)` for L-mode-like profiles:
- Profile form: `((1 - psi_hat)^alpha)^gamma`
- Returns dict with 'x' and 'y' arrays (piecewise linear)

```python
ffp_prof = create_power_flux_fun(40, 1.5, 2.0)  # FF' peaked
pp_prof = create_power_flux_fun(40, 4.0, 1.0)    # P' broad
mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)
```

## Targets and Constraints

```python
# Hard constraints
mygs.set_targets(Ip=200.E3, Ip_ratio=4.0)  # Ip_ratio ~ 1/beta_p - 1

# Soft constraints (shape)
isoflux_pts = create_isoflux(npts, R0, Z0, a, kappa, delta)
mygs.set_isoflux(isoflux_pts)  # Nx2 array of (R,Z) points on same flux surface
mygs.set_saddles(x_points)      # Nx2 array of X-point locations

# Clear constraints
mygs.set_isoflux(None)
mygs.set_saddles(None)
```

## Static Solve

```python
# Initialize psi (must be done before first solve)
mygs.init_psi(R0, Z0, a, kappa, delta)  # center, minor radius, elongation, triangularity

# Solve
err_flag = mygs.solve()  # Returns None on success
```

## Time-Dependent Solve

```python
mygs.setup_td(dt, rtol, atol)  # dt=timestep, rtol/atol for solver
sim_time = 0.0
sim_time, _, nl_its, lin_its, nretry = mygs.step_td(sim_time, dt)
# nretry >= 0 means success, < 0 means failure
```

## Extracting Results

### Global parameters
```python
stats = mygs.get_stats()
# Keys: 'Ip', 'beta_pol', 'kappa', 'kappaU', 'kappaL', 'delta', 'deltaU', 'deltaL',
#        'R_geo', 'a_geo', 'beta_tor', 'beta_n', 'q0', 'q95', 'li'
# Note: q0/q95/li may not be available for all equilibria (diverted edge cases)
```

### O-point and X-points
```python
mygs.o_point   # [R, Z] of magnetic axis
mygs.diverted  # bool: True if diverted configuration
xpts = mygs.get_xpoints()  # array of X-point locations
```

### Psi field
```python
psi = mygs.get_psi()         # normalized psi on mesh nodes
psi = mygs.get_psi(False)    # absolute psi (Wb)
mygs.set_psi(psi_array)      # set psi for restart
mygs.psi_bounds              # [psi_axis, psi_boundary]
```

### Field evaluation at arbitrary (R,Z) points
```python
B_eval = mygs.get_field_eval("B")       # returns [Br, Bt, Bz]
psi_eval = mygs.get_field_eval("psi")   # returns [psi]

# Evaluate: pass [R, Z] array
vals = B_eval.eval(np.array([0.32, 0.0]))  # -> [Br, Bt, Bz]
vals = psi_eval.eval(np.array([0.32, 0.0]))  # -> [psi]
```

Field types: `"B"`, `"psi"`, `"F"`, `"P"`, `"dPSI"`, `"dBr"`, `"dBt"`, `"dBz"`

### Mesh data
```python
mygs.r   # Nx2 array of node (R, Z) coordinates
```

## Plotting

```python
fig, ax = plt.subplots(1, 1)
mygs.plot_machine(fig, ax, coil_colormap='seismic', coil_symmap=True,
                  coil_scale=1.E-3, coil_clabel=r'$I_{coil}$ [kA]')
mygs.plot_psi(fig, ax, xpoint_color='k', vacuum_nlevels=6, plasma_nlevels=8)
mygs.plot_constraints(fig, ax, isoflux_color='tab:red', isoflux_marker='.')
mygs.plot_eddy(fig, ax, dpsi_dt=mode, colormap='seismic', symmap=True)
```

## CUTE-Specific Parameters

- Major radius R0 ~ 0.34 m (magnetic axis at ~0.34)
- Aspect ratio ~ 1.5 (spherical torus)
- 14 CS coils (CS01-CS14) at R=0.088m, Z from -0.65 to 0.65m, 100 turns each
- 14 PF coils (PF01-PF14) at varying R/Z, 48-100 turns each
- Vacuum vessel eta = 1.26e-5 ohm-m
- F0 = 0.17 (B0*R0 for toroidal field)
- Limiter is the inner VV surface
- lim_zmax = 0.38 (prevent limiting in divertor)
- Typical Ip = 200 kA, beta_p ~ 30%

## Gotchas

1. Only ONE TokaMaker instance per Python kernel: no way around this
2. `get_field_eval().eval(pt)` takes a single [R,Z] array, NOT separate R and Z arguments
3. `get_stats()` may fail after `step_td()` if the plasma boundary is poorly defined (e.g. during transients). Use `o_point` directly for tracking position.
4. `init_psi()` must be called before the first `solve()`: it sets up the initial current distribution
5. `solve()` returns `None` on success, not 0 or True
6. q0/q95 are sometimes unavailable (returned as N/A) when flux surface tracing fails at the edge: this happens for strongly shaped or diverted plasmas
7. The mesh coordinate system uses (R, Z) in meters. The `r` attribute on TokaMaker gives node positions as Nx2 array.
8. `set_saddles(None)` and `set_isoflux(None)` are used to clear constraints for free evolution
9. For time-dependent solves, remove shape constraints first, then call `setup_td()`, then loop `step_td()`
10. Coil current units are Ampere-turns. Bounds are per-turn.
