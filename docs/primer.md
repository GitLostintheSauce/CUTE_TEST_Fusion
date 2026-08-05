# Background primer

Written for someone picking this project up without a plasma physics
background. It covers the machine, the quantities that keep appearing, the
measurement problem this repository solves, and where the code sits inside all
of that.

You do not need this to run anything. You need it to explain what you are
running.

---

## 1. What a tokamak is doing

Fusion needs a plasma at a temperature no material wall can survive, so the
plasma has to be held away from the walls. A tokamak does that with magnetic
fields, exploiting the fact that charged particles spiral tightly around field
lines and move freely along them but only slowly across them.

Bend the field lines into a closed loop and particles can circulate
indefinitely without hitting anything. That is the doughnut, or torus.

**Why a purely circular field is not enough.** In a torus the field is stronger
on the inside than the outside. That gradient, plus the curvature of the field
lines, makes positive and negative charges drift in *opposite* vertical
directions. The resulting charge separation creates an electric field, and the
combination of that electric field with the magnetic field pushes the whole
plasma outward into the wall.

The fix is to make the field lines **helical** instead of purely circular, so a
particle following one spends time at the top and the bottom of the plasma. Any
charge separation gets short-circuited along the field line before it can build
up.

A tokamak gets that twist by driving a large electric current through the
plasma itself. So:

| Field | Where it comes from |
|---|---|
| **Toroidal** (the long way round) | External coils |
| **Poloidal** (the short way round) | The plasma current, plus shaping coils |

This is why **plasma current is the single most important number describing a
shot**. The plasma is not merely sitting in a magnetic bottle; it is generating
half the bottle.

---

## 2. The geometry and its vocabulary

Because a tokamak is axisymmetric (the same all the way round), you can
describe everything in a 2D slice: a **poloidal cross section**, with
coordinates $R$ (distance from the central axis) and $Z$ (height).

Nearly every plot in this project is that slice.

- **Major radius $R_0$**: distance from the machine's central axis to the
  centre of the plasma. CUTE: about 0.32 m.
- **Minor radius $a$**: the plasma's own radius. CUTE: about 0.17 m.
- **Aspect ratio $R_0/a$**: roughly 1.9 for CUTE. Conventional tokamaks are
  around 3 or more; anything near 2 is a **spherical tokamak**, shaped more
  like a cored apple than a doughnut. DIII-D, by contrast, has $R_0$ around
  1.75 m, which is why a DIII-D equilibrium can only ever be a labeled
  reference here, never a stand-in for CUTE.
- **Elongation $\kappa$**: how stretched vertically the cross section is. 1 is
  circular; CUTE targets about 1.7.
- **Triangularity $\delta$**: how much the shape is pulled into a D.

---

## 3. Flux surfaces and psi

This is the concept everything else hangs off, and it is worth slowing down
for.

Define $\psi(R, Z)$, the **poloidal magnetic flux**, as a number attached to
every point in the cross section. Its defining property:

> Magnetic field lines lie entirely within surfaces of constant $\psi$.

Picture a topographic map where $\psi$ is the elevation. Contour lines are
**flux surfaces**. A particle travels freely *along* a contour but crosses
between contours only slowly. Confinement, in one sentence.

Key features:

- **Magnetic axis**: the peak of the $\psi$ landscape, the innermost surface,
  the core of the plasma.
- **Last closed flux surface (LCFS)**, also called the **separatrix**: the
  outermost contour that still closes on itself. Inside it, plasma is
  confined; outside, field lines wander to the wall. **This contour is the
  plasma boundary.**
- **X-point**: a saddle in the $\psi$ landscape where the poloidal field
  vanishes and the separatrix crosses itself.
- **Limiter**: some machines define the boundary by physical contact with a
  surface instead. For CUTE the limiter is the inner surface of the vacuum
  vessel.

A **psi map** is simply $\psi$ evaluated on a grid. It is the complete
description of the equilibrium, which is why real equilibria are usually stored
that way.

---

## 4. The Grad-Shafranov equation

An equilibrium is a balance: outward plasma pressure against inward magnetic
force. Writing that balance for an axisymmetric plasma gives the
**Grad-Shafranov equation**, a nonlinear PDE for $\psi$:

$$\Delta^* \psi = -\mu_0 R^2 p'(\psi) - F(\psi)F'(\psi)$$

You do not need to manipulate it. You need to know what it means:

- Solving it means **finding the $\psi$ map**, and therefore the flux surfaces
  and the plasma boundary.
- $p(\psi)$ is the pressure profile and $F(\psi)$ relates to the toroidal
  field. These are **inputs**: assumptions about how pressure and current are
  distributed.
- It is nonlinear, because the right side depends on $\psi$ itself. That is why
  it is solved iteratively rather than in one step.

**Fixed boundary** means you prescribe the plasma edge and solve inside it.
**Free boundary** means the solver works out where the boundary lands, given
the coil currents and the vessel. Real machines need free boundary, and that is
what TokaMaker does.

One solve on the CUTE mesh takes about **0.26 seconds**.

---

## 5. The derived quantities

These appear constantly, and each summarizes something about the $\psi$
solution.

**$q$, the safety factor.** How many times a field line travels the long way
round the torus per single trip the short way round. Named for stability: if
$q$ falls too low, the plasma goes unstable. **$q_{95}$** is its value on the
surface enclosing 95% of the flux, near the edge.

**$\beta$ (beta).** Plasma pressure divided by magnetic pressure. It measures
how much plasma you are holding for the magnetic field you spent. Higher is
more efficient and harder to keep stable. **$\beta_{pol}$** uses the poloidal
field specifically.

**$l_i$, internal inductance.** How peaked the current profile is: whether
current is concentrated in the core or spread toward the edge. It affects
stability and how the plasma responds to control.

**Volt-seconds and flux consumption.** The central solenoid is a transformer
primary; the plasma is the secondary. Changing solenoid flux drives the loop
voltage that sustains the plasma current. There is a finite flux swing
available, which is why pulses have a maximum length, and why
`CUTE_pulse_ex.ipynb` works to maximize the starting flux.

---

## 6. The measurement problem, which is what this repository is about

You cannot photograph a plasma, and you cannot put a probe inside it. Every
diagnostic sits **outside** the vessel and reads the magnetic field the plasma
leaves behind.

Two sensor types matter here:

- **Flux loops**: wire loops encircling the machine at fixed $(R, Z)$,
  measuring $\psi$ at that point.
- **Mirnov probes**: small coils measuring the local field along their own
  orientation, so a probe at angle $\theta$ reads
  $B_R\cos\theta + B_Z\sin\theta$.

This project assumes 130 channels: 56 flux loops and 74 Mirnov probes. **That
layout is invented, not CUTE's real diagnostic set**, and every downstream
number inherits the assumption.

### Forward and inverse

- **Forward problem**: given a plasma, what would the sensors read? Direct
  computation, no ambiguity.
- **Inverse problem**: given sensor readings, what was the plasma? This is
  **equilibrium reconstruction**, and it is hard, because many plasmas produce
  similar external fields.

The classical approach guesses an equilibrium, computes what the sensors would
read, compares against what they did read, adjusts, and repeats. **EFIT** is
the decades-old standard code for this. TokaMaker is a modern open-source
solver in the same space.

Reconstruction is slow because every iteration is a full solve. That is the
opening this project's surrogate aims at.

---

## 7. How this repository fits together

Two halves that never call each other. Worth knowing precisely, because it is
an easy thing to get wrong when describing it.

### The TokaMaker half: `src/reconstruct/`

Real Grad-Shafranov work, driven through a CLI.

- `cli.py` is the only place a TokaMaker object is created: loads
  `data/CUTE_mesh.h5`, sets up the 28 coil sets, applies bounds and profiles.
- `solver.py` runs the reconstruction loop: set targets, apply shape
  constraints, `init_psi`, `solve`, iterate against measurements.
- `greens.py` builds the coil-to-sensor response matrix by solving with unit
  current in each coil.
- `efit.py` is an EFIT-style variant, `eddy.py` handles vessel eddy currents.
- `src/forward/model.py` evaluates $\psi$ and $B$ at sensor positions, turning
  a solved equilibrium into 130 numbers.

### The machine learning half: `src/ml/`

- `dataset.py` is a **reduced forward model**: the plasma is treated as a rigid
  disk of circular current filaments, evaluated analytically. Fast, and
  validated against Biot-Savart, but not a Grad-Shafranov solve.
- `mlp.py` is a neural network written from scratch in NumPy.
- `baseline.py` is a classical least-squares inversion, used as the benchmark.
- `uncertainty.py` runs an ensemble and calibrates its error bars.

### The dashboard: `src/dashboard/`

Reads pre-generated shot files. **It contains no solver code.** If someone asks
whether the dashboard runs TokaMaker, the answer is no.

---

## 8. The honest caveats, and why each exists

These are the most valuable thing in the project. Each one is a place where it
would have been easy to overclaim.

**The data is synthetic.** CUTE is not operating. There is no experimental
data anywhere in the repository, and every panel says so. For a machine that
is not yet running, synthetic data is not a shortcut; it is the only data
there is, and building the diagnostic pipeline before the hardware exists is
the point.

**The benchmark is against the same reduced model.** The surrogate is roughly
13,000x faster than a least-squares inversion *of the reduced model it was
trained on*, not faster than a Grad-Shafranov solve. Comparing against
TokaMaker would mostly measure reduced physics against full physics, which is a
different and less honest claim.

**The surrogate predicts $I_p$, $R_0$, $Z_0$, $a$ and deliberately not
$q_{95}$, $\beta_{pol}$, $l_i$.** In the reduced dataset those three are
computed from $I_p$ by formula, so predicting them would be relearning
arithmetic. Measured on 2000 real Grad-Shafranov equilibria, $I_p$ explains
42% of $q_{95}$'s variance, 7% of $\beta_{pol}$'s and 6% of $l_i$'s, so on that
dataset they become worth predicting. Stored energy stays excluded at 97%.

**The dashboard's flux contours are labeled illustrative.** The stored shots
keep the boundary and scalars but not a $\psi$ map, so those contours are
inferred from boundary shape rather than solved.

**Coverage says 52%, not 77%.** OFT does not import on the CI runner, so 48
solver tests skip there. The badge reports what CI can verify.

**The noise model is invented.** 2% independent Gaussian noise per channel.
Real magnetic diagnostics drift, correlate between neighbours, and spike. Any
error bar is conditional on that assumption.

---

## 9. The surrounding tools

**Open Fusion Toolkit (OFT)**, and **TokaMaker** within it: the free-boundary
Grad-Shafranov solver this project is built on. Maintained by Chris Hansen.
Ships example notebooks including several for CUTE.

**EFIT**: the legacy standard reconstruction code. Its `g-file` format is
universal, so reading g-files stays useful even though the code itself is
dated.

**TORAX**: an open-source core transport simulator from Google DeepMind,
written in JAX. Where TokaMaker computes the equilibrium at an instant, TORAX
evolves temperature, density and current *through time*. They are
complementary, and a coupled workflow exists in the OFT examples. Being written
in JAX makes it differentiable, which is what makes gradient-based optimization
possible.

---

## 10. Questions you should be able to answer

If these are comfortable, you can hold a conversation about this project.

1. Why does a tokamak need plasma current at all?
2. What is a flux surface, and why does it confine anything?
3. What does solving Grad-Shafranov actually produce?
4. What is the difference between the forward and inverse problem here?
5. Why is the surrogate benchmarked against least-squares rather than
   TokaMaker?
6. Why does the surrogate refuse to predict $q_{95}$ on the reduced dataset?
7. Does the dashboard use TokaMaker?
8. What would break if a dataset were labeled with the requested plasma current
   rather than the achieved one?

Answers to 5, 6, 7 and 8, in short: because the surrogate learned the reduced
model and a fair benchmark solves the same problem on the same physics; because
$q_{95}$ is computed from $I_p$ there by formula; no, it reads pre-generated
files; and the network would be asked to predict a number its inputs do not
contain, putting a floor under its error and quietly miscalibrating the
uncertainty estimates.
