# CUTE Magnetic Diagnostic Pipeline & Equilibrium Reconstruction

## Overview

An end-to-end pipeline that ingests raw magnetic sensor data from the CUTE spherical tokamak (56+ flux loops, 74+ Mirnov probes), processes it, reconstructs plasma equilibria via OFT's TokaMaker, and presents results through a live dashboard designed for remote operation.

---

## DAG

```
Phase 1  ───►  Phase 2  ───►  Phase 3a  ───►  Phase 5  ───►  Phase 6a  ───►  Phase 7
                              Phase 3b  ───►  Phase 5         Phase 6b  ───►  Phase 7
                              Phase 3c  ───►  Phase 5
                                                              Phase 4   (anytime after Phase 2)
```

- Phases 3a/3b/3c are independent of each other (parallel)
- Phases 6a/6b are independent of each other (parallel)
- Phase 4 can be done anytime after Phase 2

---

## Phase 1: Bootstrap — Dev Environment & OFT Setup

**Depends on:** nothing
**Goal:** Go from zero to a working OFT install with a green CI pipeline.
**One-shot scope:** ~2-3 hours

### Do this

1. Create Python environment (>=3.10) with `pyproject.toml`, pin core deps:
   - `OpenFUSIONToolkit`, `numpy`, `scipy`, `h5py`, `pydantic`, `plotly`, `dash`, `pytest`
2. Install OFT from GitHub release binaries (macOS arm64) or build from source
3. Run OFT's built-in test suite — confirm it passes
4. Run at least one TokaMaker example notebook end-to-end
5. Create the project directory structure:

```
src/
├── forward/          # Phase 3a
├── signal/           # Phase 3b
├── store/            # Phase 3c
├── reconstruct/      # Phase 5
└── dashboard/        # Phase 6a
tests/
notebooks/
data/
  ├── raw/
  ├── processed/
  └── synthetic/
```

6. Set up GitHub Actions CI: lint (ruff), type-check (mypy), run pytest on push

### Done when

- [x] `oft --version` works
- [x] TokaMaker example notebook runs without errors
- [x] `git push` triggers CI and it's green (empty test suite is fine)

---

## Phase 2: Learn TokaMaker — Reference CUTE Equilibrium

**Depends on:** Phase 1
**Goal:** Produce a credible CUTE-like plasma equilibrium and understand the TokaMaker API well enough to script it.
**One-shot scope:** ~3-4 hours

### Do this

1. Work through all existing TokaMaker examples/notebooks in the OFT repo
2. Research CUTE geometry from published sources: major radius (~0.3m), aspect ratio (~1.5), coil positions, vessel shape
3. Build a CUTE-geometry mesh (triangular, 2D axisymmetric) for TokaMaker
4. Run a **static** Grad-Shafranov solve → reference equilibrium
5. Run a **time-dependent** solve: current ramp-up → flat-top → ramp-down
6. Extract and plot: flux surfaces, plasma current (Ip), q-profile, Shafranov shift, beta
7. Write up TokaMaker API notes in `notebooks/tokamaker_api_notes.md`:
   - How to set up a mesh
   - How to configure coils and boundary conditions
   - How to run static vs. time-dependent solves
   - How to extract field values at arbitrary points
   - Any gotchas or undocumented behavior

### Done when

- [x] A notebook `notebooks/cute_reference_equilibrium.ipynb` produces annotated flux surface plots
- [x] API notes exist and cover all the points above
- [x] You can programmatically: create mesh → solve → extract B-field at a point

---

## Phase 3a: Synthetic Sensor Forward Model

**Depends on:** Phase 2
**Goal:** Given a TokaMaker equilibrium, produce what every sensor *would* measure.
**One-shot scope:** ~3-4 hours

### Do this

1. Define sensor geometry in a config file (`config/sensors.toml`):
   - Positions (R, Z) and orientations for all 56 flux loops
   - Positions (R, Z) and orientations for all 74 Mirnov probes
   - Use CUTE design docs, or best estimates from published papers
2. Implement in `src/forward/`:
   ```python
   def flux_loop(equilibrium, sensor_pos: tuple[float, float]) -> float
       """Poloidal flux at sensor (R, Z) location."""

   def mirnov_probe(equilibrium, sensor_pos, sensor_orient) -> float
       """B-field component at probe location along probe orientation."""

   def full_diagnostic_set(equilibrium, sensor_config) -> SensorFrame
       """All sensors, vectorized. Returns timestamped dataframe."""
   ```
3. Add a configurable noise model in `src/forward/noise.py`:
   - Gaussian white noise (configurable sigma)
   - 60 Hz sinusoidal pickup (configurable amplitude)
   - Random channel dropout (configurable probability)
4. Generate a full synthetic shot dataset from the Phase 2 reference discharge:
   - Time series of all sensor signals across ramp-up/flat-top/ramp-down
   - Save to `data/synthetic/shot_001.h5`
5. Write unit tests:
   - Zero-noise forward model on a Solov'ev analytic equilibrium must match analytic B-field to **<1% error**
   - Noise model produces correct statistical properties (mean, std, frequency content)

### Done when

- [x] `forward.full_diagnostic_set()` produces realistic waveforms for the reference shot
- [x] Solov'ev analytic validation test passes
- [x] `data/synthetic/shot_001.h5` exists with clean + noisy variants

---

## Phase 3b: Signal Processing Library

**Depends on:** Phase 2 (understand signal characteristics)
**Goal:** Take raw sensor signals in, get clean calibrated data out.
**One-shot scope:** ~3 hours

### Do this

1. Define data classes in `src/signal/types.py`:
   ```python
   @dataclass
   class RawSignal:
       timestamps: np.ndarray    # seconds
       values: np.ndarray        # raw ADC units
       channel_id: str
       sensor_type: Literal["flux_loop", "mirnov"]
       metadata: dict

   @dataclass
   class ProcessedSignal:
       timestamps: np.ndarray
       values: np.ndarray        # physical units (Wb or T)
       channel_id: str
       sensor_type: Literal["flux_loop", "mirnov"]
       processing_log: list[str] # what was applied
   ```
2. Implement processing steps in `src/signal/`:
   - `calibrate.py` — gain correction, offset subtraction per channel (params from config)
   - `filters.py` — bandpass (Butterworth 4th order, zero-phase `filtfilt`), configurable cutoffs
   - `filters.py` — 60 Hz notch filter + harmonics (120, 180 Hz)
   - `outliers.py` — dropout detection (threshold-based), linear interpolation over gaps
   - `integrate.py` — drift correction for Mirnov probes (numeric integration of dB/dt → B, with baseline subtraction)
3. Compose into top-level function:
   ```python
   def process_shot(raw_signals: list[RawSignal], config: ProcessingConfig) -> list[ProcessedSignal]
   ```
4. All filter parameters configurable via `config/processing.toml`
5. Write unit tests using Phase 3a synthetic data:
   - Inject known noise → process → compare to ground truth
   - Verify SNR improvement
   - Verify phase preservation (cross-correlation > 0.99)

### Done when

- [x] Processing a noisy synthetic shot recovers clean signals within **5% of ground truth**
- [x] Config file controls all filter parameters
- [x] Tests pass

---

## Phase 3c: Data Schema & Storage Layer

**Depends on:** Phase 2 (know what data shapes look like)
**Goal:** Structured, validated storage so every other module has a clean interface to read/write data.
**One-shot scope:** ~2 hours

### Do this

1. Define schemas in `src/store/schemas.py` using Pydantic v2:
   ```python
   class ShotMetadata(BaseModel):
       shot_number: int
       timestamp: datetime
       coil_currents: dict[str, float]  # Amps
       gas_pressure: float              # Torr
       operator_notes: str = ""

   class SignalMetadata(BaseModel):
       channel_id: str
       sensor_type: Literal["flux_loop", "mirnov"]
       position_r: float  # meters
       position_z: float  # meters
       orientation: float  # radians

   class EquilibriumResult(BaseModel):
       plasma_current: float    # Amps
       q95: float
       beta_poloidal: float
       internal_inductance: float
       boundary_r: list[float]
       boundary_z: list[float]
   ```
2. Implement HDF5 backend in `src/store/hdf5.py`:
   - `save_shot(path, metadata, raw_signals, processed_signals=None, equilibrium=None)`
   - `load_shot(path) -> Shot` (with lazy loading for large arrays)
   - One HDF5 file per shot: `/meta`, `/raw/{channel_id}`, `/processed/{channel_id}`, `/equilibrium`
3. Implement shot index:
   - `index(directory) -> pd.DataFrame` — scan all `.h5` files, return summary table
4. Write unit tests:
   - Round-trip: save → load → compare (exact match)
   - Schema validation: malformed data raises `ValidationError`
   - Index returns correct count and metadata

### Done when

- [x] A synthetic shot can be saved, loaded, and indexed
- [x] Pydantic models are the single source of truth for all data shapes
- [x] Tests pass

---

## Phase 4: OFT Upstream Contributions

**Depends on:** Phase 2 (can start filing issues/PRs anytime after you've used OFT)
**Goal:** Give back to the project. This is ongoing, not a single sitting.
**One-shot scope:** ~2 hours per contribution

### Do this (incrementally, throughout the project)

1. File GitHub issues for any bugs, confusing API, or missing docs you hit during Phase 2+
2. Improve TokaMaker Python examples based on your Phase 2 learnings
3. Contribute a CUTE-geometry example to OFT's example gallery
4. If Python bindings are lacking (e.g., can't easily evaluate B-field at a point), propose/implement improvements
5. Add or improve docstrings where documentation was missing

### Done when

- [x] At least 2 issues filed on OpenFUSIONToolkit/OpenFUSIONToolkit
- [x] At least 1 PR submitted (example, docs, or code fix)

---

## Phase 5: Equilibrium Reconstruction Pipeline (Core Deliverable)

**Depends on:** Phase 3a + Phase 3b + Phase 3c (all three)
**Goal:** Given processed magnetic measurements, automatically reconstruct the plasma equilibrium.
**One-shot scope:** ~6-8 hours (largest phase — can be split across two sessions)

### Session A: Core reconstruction loop (~3-4 hours)

1. Implement measurement-to-constraint mapping in `src/reconstruct/constraints.py`:
   - Convert processed flux loop signals → flux values at sensor (R, Z) locations
   - Convert processed Mirnov signals → B-field components at probe locations
2. Implement the iterative reconstruction loop in `src/reconstruct/solver.py`:
   ```
   function fit_equilibrium(processed_signals, sensor_config, mesh):
       guess = vacuum_field(mesh)          # or previous time-step
       for iteration in range(max_iter):
           equilibrium = tokamaker_solve(mesh, guess, constraints)
           synthetic = forward.full_diagnostic_set(equilibrium, sensor_config)
           residual = measured - synthetic
           if norm(residual) < tolerance:
               break
           update constraints / current profile from residual
       return equilibrium, diagnostics
   ```
3. Wire up: `reconstruct.fit_equilibrium(processed_signals, sensor_config, mesh) -> Equilibrium`
4. Test on synthetic data (zero noise): reconstructed equilibrium must exactly match input

### Session B: Time-series, CLI, validation (~3-4 hours)

5. Implement time-series reconstruction in `src/reconstruct/timeseries.py`:
   - Loop over time slices
   - Warm-start each slice from previous solution
   - Output: list of `EquilibriumResult` + per-slice diagnostics
6. Output residual diagnostics: per-sensor fit quality, chi-squared, condition number
7. Build CLI entry point:
   ```
   cute-reconstruct --shot data/synthetic/shot_001.h5 --output results.h5
   ```
8. Validation on noisy synthetic data:
   - Plasma boundary position error **< 1 cm**
   - Ip recovery **< 2% error**
   - q95 recovery **< 10% error**
9. Performance profiling: target **< 2 seconds per time-slice**

### Done when

- [x] Synthetic round-trip test passes all accuracy targets
- [x] CLI command works end-to-end
- [x] Tests pass

---

## Phase 6a: Dashboard & Visualization

**Depends on:** Phase 5
**Goal:** Interactive web dashboard for remote shot review and monitoring.
**One-shot scope:** ~4-5 hours

### Do this

1. Set up Plotly Dash app in `src/dashboard/app.py`
2. **Shot browser page** — table from `store.index()`, click a row to load that shot
3. **Signal viewer** — time-series plots of raw vs. processed signals, dropdown to select sensor channel
4. **Equilibrium viewer** — 2D contour plot of flux surfaces with:
   - Plasma boundary (thick line)
   - Limiter/vessel wall
   - Sensor positions (dots, colored by type)
   - Time slider to scrub through the discharge
5. **Parameter timeline** — Ip, q95, beta, li plotted vs. time for the loaded shot
6. **Sim vs. experiment** — overlay forward-model synthetic signals on measured signals, show residual below
7. **Live mode** — watch `data/` directory with `watchdog`, auto-process new shots and refresh
8. **Export** — download any plot as PNG, download data as CSV or HDF5
9. Basic token-based auth gate for remote access

### Done when

- [x] `python -m src.dashboard.app` launches a working dashboard on localhost
- [x] All views render correctly for a synthetic shot
- [x] A person unfamiliar with the code can understand the plasma behavior from the dashboard

---

## Phase 6b: Validation & Benchmarks

**Depends on:** Phase 5
**Goal:** Quantitative confidence bounds on the reconstruction pipeline.
**One-shot scope:** ~3-4 hours

### Do this

1. **Analytic benchmark** — Solov'ev equilibrium with exact sensor values → reconstruction must match analytically (no iteration needed, residual ≈ 0)
2. **Noise sweep** — reconstruct the same equilibrium at SNR = ∞, 40dB, 20dB, 10dB
   - Plot: reconstruction error vs. SNR for each key parameter (Ip, q95, boundary R)
3. **Sensor dropout study** — systematically remove 10%, 25%, 50% of sensors (random draws, N=20 each)
   - Plot: error vs. fraction of sensors removed
   - Identify which individual sensors are most critical (leave-one-out)
4. **Mesh convergence study** — run at 3-4 mesh resolutions, confirm key outputs converge
5. Write validation report: `notebooks/validation_report.ipynb`
   - All figures reproducible by re-running the notebook
   - Summary table of accuracy under each condition
   - Documented failure modes and known limitations

### Done when

- [x] Validation notebook runs end-to-end and produces all figures
- [x] Known accuracy limits are quantified and documented
- [x] At least one surprising finding is noted (there always is one)

---

## Phase 7: Documentation & Handoff

**Depends on:** Phase 6a + Phase 6b
**Goal:** A new student can pick this up and run with it.
**One-shot scope:** ~2-3 hours

### Do this

1. Write top-level `README.md`:
   - What this is (one paragraph)
   - Quickstart: install, generate synthetic data, run reconstruction, launch dashboard
   - Architecture diagram (ASCII or Mermaid)
   - Link to spec.md for full context
2. Write `docs/architecture.md`:
   - Data flow diagram: raw signals → processing → reconstruction → dashboard
   - Module responsibilities and boundaries
   - Key design decisions and why
3. Write `docs/operator_guide.md`:
   - How to process a new shot
   - How to launch and use the dashboard
   - How to add a new sensor type
   - Troubleshooting common issues
4. Record a 2-3 minute demo (GIF or video) showing the full pipeline
5. Clean all notebooks: clear outputs, re-run, confirm they pass
6. Tag `v0.1.0` release on GitHub

### Done when

- [x] A new student can clone → install → process a synthetic shot → view dashboard in **< 30 minutes**
- [x] CI passes on the tagged release
- [x] README, architecture doc, and operator guide all exist

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python 3.10+ | OFT Python bindings, scientific ecosystem |
| Solver | OFT / TokaMaker | Project requirement, Fortran-backed FEM |
| Signal processing | SciPy | Standard, well-tested DSP |
| Storage | HDF5 via h5py | Fusion community standard, large time-series |
| Schema validation | Pydantic v2 | Fast, type-safe, good errors |
| Dashboard | Plotly Dash | Scientific viz, pure Python |
| Testing | pytest | Fixtures, parametrize |
| CI | GitHub Actions | Free, good caching |
| Config | TOML | Python-native, human-readable |

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| CUTE geometry details unavailable | Can't build accurate mesh | Use published papers + HBT-EP as proxy; refine when specs arrive |
| TokaMaker Python API insufficient for iterative reconstruction | Blocks Phase 5 | Identify gaps in Phase 2 early; fall back to subprocess calls to OFT CLI |
| No real experimental data during project | Can't validate against experiment | Synthetic data pipeline (Phase 3a) is the fallback; design for easy swap-in |
| Reconstruction doesn't converge | Core deliverable at risk | Start with simple current profiles; add complexity incrementally |
| OFT build breaks on macOS | Blocks everything | Use pre-built binaries; pin to a known-good release tag |

## Time Estimate (rough)

| Phase | Effort |
|-------|--------|
| 1. Bootstrap | ~2-3 hrs |
| 2. Learn TokaMaker | ~3-4 hrs |
| 3a. Forward Model | ~3-4 hrs |
| 3b. Signal Processing | ~3 hrs |
| 3c. Data Storage | ~2 hrs |
| 4. OFT Contributions | ~2 hrs (ongoing) |
| 5. Reconstruction | ~6-8 hrs (2 sessions) |
| 6a. Dashboard | ~4-5 hrs |
| 6b. Validation | ~3-4 hrs |
| 7. Docs & Handoff | ~2-3 hrs |
| **Total** | **~30-38 hrs** |
