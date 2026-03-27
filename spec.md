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

## Acceptance Criteria Rules

Every phase has an **Acceptance Gate** section. A phase is NOT complete until every criterion is met. Criteria are categorized:

- **`[AUTO]`** — verified by an automated test (`pytest`, CI, or a script). Must be a passing test in the test suite before the phase can close.
- **`[SCRIPT]`** — verified by running a specific command and checking the output. The exact command is given.
- **`[HUMAN]`** — requires human judgment. A specific question is provided that must be answered "yes."

**No phase may be merged to `main` until all its acceptance criteria pass.**

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

### Acceptance Gate

| ID | Type | Criterion | Verification |
|----|------|-----------|--------------|
| 1.1 | `[SCRIPT]` | OFT is installed and importable | `python -c "import OpenFUSIONToolkit; print(OpenFUSIONToolkit.__version__)"` exits 0 and prints a version string |
| 1.2 | `[SCRIPT]` | TokaMaker is functional | `python -c "from OpenFUSIONToolkit.TokaMaker import TokaMaker; print('ok')"` exits 0 |
| 1.3 | `[SCRIPT]` | Project installs cleanly | `pip install -e ".[dev]"` exits 0 with no errors |
| 1.4 | `[SCRIPT]` | Directory structure exists | `ls src/forward src/signal src/store src/reconstruct src/dashboard tests notebooks data/raw data/processed data/synthetic` exits 0 |
| 1.5 | `[SCRIPT]` | Linting passes | `ruff check src/ tests/` exits 0 |
| 1.6 | `[SCRIPT]` | Type checking passes | `mypy src/` exits 0 |
| 1.7 | `[SCRIPT]` | Test suite runs (even if empty) | `pytest tests/ --tb=short` exits 0 |
| 1.8 | `[SCRIPT]` | CI pipeline is configured | `cat .github/workflows/ci.yml` exits 0 and contains `ruff`, `mypy`, and `pytest` steps |
| 1.9 | `[SCRIPT]` | CI passes on remote | `gh run list --limit 1 --json conclusion -q '.[0].conclusion'` returns `"success"` |

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

### Acceptance Gate

| ID | Type | Criterion | Verification |
|----|------|-----------|--------------|
| 2.1 | `[SCRIPT]` | Reference equilibrium notebook exists | `test -f notebooks/cute_reference_equilibrium.ipynb` exits 0 |
| 2.2 | `[SCRIPT]` | Notebook executes without error | `jupyter nbconvert --to notebook --execute notebooks/cute_reference_equilibrium.ipynb --output /dev/null` exits 0 |
| 2.3 | `[SCRIPT]` | API notes exist and cover required topics | `grep -l "mesh" notebooks/tokamaker_api_notes.md && grep -l "coil" notebooks/tokamaker_api_notes.md && grep -l "time-dependent" notebooks/tokamaker_api_notes.md && grep -l "gotcha" notebooks/tokamaker_api_notes.md` — all exit 0 |
| 2.4 | `[AUTO]` | Static solve produces valid equilibrium | `tests/test_phase2.py::test_static_solve_produces_valid_equilibrium` — asserts Ip > 0, q95 > 1, flux surface count > 0 |
| 2.5 | `[AUTO]` | Time-dependent solve runs to completion | `tests/test_phase2.py::test_time_dependent_solve_completes` — asserts solution has > 10 time steps spanning ramp-up through ramp-down |
| 2.6 | `[AUTO]` | B-field extraction works at arbitrary point | `tests/test_phase2.py::test_bfield_extraction` — extracts (Br, Bz) at 3 random (R,Z) points inside plasma, asserts non-NaN and physically reasonable magnitude (0 < \|B\| < 2 T) |
| 2.7 | `[HUMAN]` | Flux surface plot looks physically reasonable | Open `notebooks/cute_reference_equilibrium.ipynb` — do the flux surfaces show closed nested contours with a magnetic axis near the expected major radius (~0.3m)? |

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
5. Write unit tests (see acceptance gate below)

### Acceptance Gate

| ID | Type | Criterion | Verification |
|----|------|-----------|--------------|
| 3a.1 | `[SCRIPT]` | Sensor config exists and is well-formed | `python -c "import tomllib; d=tomllib.load(open('config/sensors.toml','rb')); assert len(d['flux_loops']) >= 56; assert len(d['mirnov_probes']) >= 74; print('ok')"` exits 0 |
| 3a.2 | `[AUTO]` | Solov'ev analytic validation — flux loops | `tests/test_forward.py::test_flux_loop_solovev` — Compute Solov'ev equilibrium analytically, evaluate `flux_loop()` at 10 sensor locations, assert relative error < 1% vs. analytic poloidal flux |
| 3a.3 | `[AUTO]` | Solov'ev analytic validation — Mirnov probes | `tests/test_forward.py::test_mirnov_solovev` — Same as above for `mirnov_probe()` vs. analytic B-field components, relative error < 1% |
| 3a.4 | `[AUTO]` | Full diagnostic set returns correct shape | `tests/test_forward.py::test_full_diagnostic_set_shape` — Output has 130 columns (56 + 74 sensors) and N_timesteps rows matching input equilibrium time series |
| 3a.5 | `[AUTO]` | Noise model: white noise has correct statistics | `tests/test_forward.py::test_white_noise_statistics` — Generate 10,000 samples with sigma=0.1, assert `abs(mean) < 0.01` and `abs(std - 0.1) < 0.01` |
| 3a.6 | `[AUTO]` | Noise model: 60 Hz pickup has correct frequency | `tests/test_forward.py::test_60hz_pickup_frequency` — Apply FFT to noisy signal, assert peak in power spectrum is at 60 ± 1 Hz |
| 3a.7 | `[AUTO]` | Noise model: dropout produces NaN at expected rate | `tests/test_forward.py::test_dropout_rate` — With dropout_prob=0.05, generate 10,000 samples, assert NaN fraction is 0.05 ± 0.02 |
| 3a.8 | `[SCRIPT]` | Synthetic shot file exists | `python -c "import h5py; f=h5py.File('data/synthetic/shot_001.h5','r'); assert 'clean' in f; assert 'noisy' in f; print('ok')"` exits 0 |
| 3a.9 | `[HUMAN]` | Synthetic waveforms look realistic | Plot 3 flux loop and 3 Mirnov channels from the synthetic shot — do they show expected time evolution (ramp-up, flat-top, ramp-down) with no obvious artifacts? |

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
5. Write unit tests (see acceptance gate below)

### Acceptance Gate

| ID | Type | Criterion | Verification |
|----|------|-----------|--------------|
| 3b.1 | `[AUTO]` | Calibration applies gain and offset correctly | `tests/test_signal.py::test_calibration` — Input signal with known gain=2.0, offset=0.5 → output = (input - 0.5) / 2.0, assert exact match |
| 3b.2 | `[AUTO]` | Bandpass filter removes out-of-band content | `tests/test_signal.py::test_bandpass` — Input: 100 Hz sine + 5000 Hz sine. Bandpass 50-500 Hz. Assert output power at 5000 Hz is < 1% of input power at 5000 Hz. Assert output power at 100 Hz is > 95% of input power at 100 Hz |
| 3b.3 | `[AUTO]` | Notch filter removes 60 Hz and harmonics | `tests/test_signal.py::test_notch_filter` — Input: broadband signal + 60/120/180 Hz tones. After notch, assert power at 60, 120, 180 Hz each reduced by > 20 dB |
| 3b.4 | `[AUTO]` | Bandpass filter preserves phase | `tests/test_signal.py::test_phase_preservation` — Input: known sine wave. After bandpass (in-band). Cross-correlation between input and output > 0.99 |
| 3b.5 | `[AUTO]` | Dropout detection finds and fills gaps | `tests/test_signal.py::test_dropout_interpolation` — Input: clean signal with 5% of samples set to NaN. After processing, assert no NaN values remain and interpolated values within 10% of original clean values |
| 3b.6 | `[AUTO]` | Integrator drift correction recovers B from dB/dt | `tests/test_signal.py::test_integration_drift` — Input: dB/dt of a known B(t) waveform plus linear drift. After integration + drift correction, assert recovered B(t) within 5% of true B(t) |
| 3b.7 | `[AUTO]` | End-to-end: noisy synthetic → processed ≈ clean | `tests/test_signal.py::test_end_to_end_recovery` — Take Phase 3a noisy shot, process it, compare each channel to the clean ground truth. Assert RMS error < 5% of signal RMS for at least 90% of channels |
| 3b.8 | `[SCRIPT]` | Config file controls filter parameters | `python -c "import tomllib; d=tomllib.load(open('config/processing.toml','rb')); assert 'bandpass' in d; assert 'low_cutoff' in d['bandpass']; assert 'high_cutoff' in d['bandpass']; assert 'notch' in d; print('ok')"` exits 0 |
| 3b.9 | `[AUTO]` | Processing log records what was applied | `tests/test_signal.py::test_processing_log` — After `process_shot()`, assert each `ProcessedSignal.processing_log` is a non-empty list containing at least `["calibrate", "bandpass", "notch"]` |

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
4. Write unit tests (see acceptance gate below)

### Acceptance Gate

| ID | Type | Criterion | Verification |
|----|------|-----------|--------------|
| 3c.1 | `[AUTO]` | Schema validates correct data | `tests/test_store.py::test_schema_valid_data` — Construct `ShotMetadata`, `SignalMetadata`, `EquilibriumResult` with valid data → no exceptions |
| 3c.2 | `[AUTO]` | Schema rejects invalid data | `tests/test_store.py::test_schema_rejects_invalid` — `ShotMetadata(shot_number="not_an_int", ...)` raises `ValidationError`. `SignalMetadata(sensor_type="invalid_type", ...)` raises `ValidationError`. At least 3 invalid-input cases tested |
| 3c.3 | `[AUTO]` | Round-trip save/load — metadata | `tests/test_store.py::test_roundtrip_metadata` — Save a shot, load it back, assert `loaded.metadata == original.metadata` (field-by-field equality) |
| 3c.4 | `[AUTO]` | Round-trip save/load — signal arrays | `tests/test_store.py::test_roundtrip_signals` — Save a shot with 5 raw signal channels, load it back, assert `np.array_equal(loaded.raw[ch].values, original.raw[ch].values)` for each channel |
| 3c.5 | `[AUTO]` | Round-trip save/load — equilibrium | `tests/test_store.py::test_roundtrip_equilibrium` — Save a shot with equilibrium data, load it back, assert all scalar and array fields match |
| 3c.6 | `[AUTO]` | HDF5 file structure is correct | `tests/test_store.py::test_hdf5_structure` — Save a shot, open with h5py, assert groups `/meta`, `/raw`, `/processed`, `/equilibrium` exist |
| 3c.7 | `[AUTO]` | Index returns correct summary | `tests/test_store.py::test_index` — Save 3 shots to a temp dir, call `index()`, assert DataFrame has 3 rows and columns include `shot_number`, `timestamp`, `plasma_current` |
| 3c.8 | `[AUTO]` | Load with missing optional data works | `tests/test_store.py::test_load_partial` — Save a shot with only raw signals (no processed, no equilibrium), load it back, assert `loaded.processed is None` and `loaded.equilibrium is None` |
| 3c.9 | `[SCRIPT]` | Pydantic models are importable as public API | `python -c "from src.store.schemas import ShotMetadata, SignalMetadata, EquilibriumResult; print('ok')"` exits 0 |

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

### Acceptance Gate

| ID | Type | Criterion | Verification |
|----|------|-----------|--------------|
| 4.1 | `[SCRIPT]` | At least 2 issues filed | `gh issue list --repo OpenFUSIONToolkit/OpenFUSIONToolkit --author @me --json number -q 'length'` returns >= 2 |
| 4.2 | `[SCRIPT]` | At least 1 PR submitted | `gh pr list --repo OpenFUSIONToolkit/OpenFUSIONToolkit --author @me --json number -q 'length'` returns >= 1 |
| 4.3 | `[HUMAN]` | Issues are substantive, not filler | Review your filed issues — does each describe a real bug, usability problem, or missing documentation that another user would benefit from? |

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
8. Validation on noisy synthetic data (accuracy targets below)
9. Performance profiling: target < 2 seconds per time-slice

### Acceptance Gate

| ID | Type | Criterion | Verification |
|----|------|-----------|--------------|
| 5.1 | `[AUTO]` | Zero-noise round-trip: flux loops | `tests/test_reconstruct.py::test_zero_noise_flux_loop_roundtrip` — Generate clean synthetic signals from known equilibrium → reconstruct → compare forward model output of reconstruction to input signals. Assert max absolute error < 0.1% of signal range for all flux loops |
| 5.2 | `[AUTO]` | Zero-noise round-trip: Ip recovery | `tests/test_reconstruct.py::test_zero_noise_ip_recovery` — Reconstruct from clean synthetic data, assert `abs(reconstructed_Ip - true_Ip) / true_Ip < 0.01` (1% error) |
| 5.3 | `[AUTO]` | Zero-noise round-trip: boundary recovery | `tests/test_reconstruct.py::test_zero_noise_boundary_recovery` — Assert max Euclidean distance between reconstructed and true plasma boundary points < 0.5 cm |
| 5.4 | `[AUTO]` | Noisy reconstruction: Ip recovery | `tests/test_reconstruct.py::test_noisy_ip_recovery` — Reconstruct from noisy synthetic data (SNR=20dB), assert `abs(reconstructed_Ip - true_Ip) / true_Ip < 0.02` (2% error) |
| 5.5 | `[AUTO]` | Noisy reconstruction: q95 recovery | `tests/test_reconstruct.py::test_noisy_q95_recovery` — Same noisy input, assert `abs(reconstructed_q95 - true_q95) / true_q95 < 0.10` (10% error) |
| 5.6 | `[AUTO]` | Noisy reconstruction: boundary recovery | `tests/test_reconstruct.py::test_noisy_boundary_recovery` — Same noisy input, assert max boundary error < 1.0 cm |
| 5.7 | `[AUTO]` | Convergence: iteration count is bounded | `tests/test_reconstruct.py::test_convergence` — Assert reconstruction converges in < 50 iterations for the reference equilibrium |
| 5.8 | `[AUTO]` | Residual diagnostics are populated | `tests/test_reconstruct.py::test_residual_diagnostics` — After reconstruction, assert `diagnostics.chi_squared` is a float > 0, `diagnostics.per_sensor_residual` has 130 entries, `diagnostics.condition_number` is a float > 0 |
| 5.9 | `[AUTO]` | Time-series reconstruction warm-starts correctly | `tests/test_reconstruct.py::test_timeseries_warmstart` — Reconstruct 5 consecutive time slices. Assert slice 2-5 each converge in fewer iterations than slice 1 (warm-start benefit) |
| 5.10 | `[AUTO]` | Time-series output has correct structure | `tests/test_reconstruct.py::test_timeseries_output_structure` — Assert output is a list of N `EquilibriumResult` objects matching the N input time slices, each with all required fields populated |
| 5.11 | `[SCRIPT]` | CLI runs end-to-end | `cute-reconstruct --shot data/synthetic/shot_001.h5 --output /tmp/test_results.h5 && python -c "import h5py; f=h5py.File('/tmp/test_results.h5','r'); assert 'equilibrium' in f; print('ok')"` exits 0 |
| 5.12 | `[SCRIPT]` | Performance: < 2s per time-slice | `python -c "import time; from src.reconstruct import fit_equilibrium; [run benchmark on reference equilibrium]; assert elapsed/n_slices < 2.0"` — average wall time per slice < 2.0 seconds |
| 5.13 | `[HUMAN]` | Reconstructed flux surfaces look correct | Plot reconstructed equilibrium next to ground truth for 3 time slices (ramp-up, flat-top, ramp-down) — do the flux surface shapes, axis position, and boundary match qualitatively? |

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

### Acceptance Gate

| ID | Type | Criterion | Verification |
|----|------|-----------|--------------|
| 6a.1 | `[SCRIPT]` | App starts without error | `timeout 10 python -m src.dashboard.app --port 8099 &; sleep 3; curl -s -o /dev/null -w '%{http_code}' http://localhost:8099` returns `200`; kill background process |
| 6a.2 | `[AUTO]` | Shot browser returns correct data | `tests/test_dashboard.py::test_shot_browser_data` — Mock 3 shots in store, call the shot browser callback, assert returned table has 3 rows with correct shot numbers |
| 6a.3 | `[AUTO]` | Signal viewer callback returns valid figure | `tests/test_dashboard.py::test_signal_viewer_callback` — Call signal viewer callback with a valid shot and channel ID, assert returned object is a `plotly.graph_objects.Figure` with at least 1 trace |
| 6a.4 | `[AUTO]` | Equilibrium viewer callback returns valid figure | `tests/test_dashboard.py::test_equilibrium_viewer_callback` — Call equilibrium viewer callback with a valid shot and time index, assert returned Figure has contour trace + boundary trace + sensor scatter trace |
| 6a.5 | `[AUTO]` | Parameter timeline callback returns valid figure | `tests/test_dashboard.py::test_parameter_timeline_callback` — Call with valid shot, assert Figure has traces for Ip, q95, beta, li (4 traces) |
| 6a.6 | `[AUTO]` | Sim-vs-experiment callback returns valid figure | `tests/test_dashboard.py::test_sim_vs_experiment_callback` — Call with valid shot and channel, assert Figure has 3 traces (measured, synthetic, residual) |
| 6a.7 | `[SCRIPT]` | Export produces valid files | Start app, use `requests` to hit export endpoint → assert returned file is valid PNG (check magic bytes) or valid HDF5 |
| 6a.8 | `[AUTO]` | Auth gate rejects unauthenticated requests | `tests/test_dashboard.py::test_auth_gate` — Request without token returns 401/403. Request with valid token returns 200 |
| 6a.9 | `[HUMAN]` | Dashboard is usable by a non-expert | Have someone unfamiliar with the project open the dashboard with a synthetic shot loaded. Ask them: "Can you tell me what the plasma current was during flat-top?" and "Can you tell me roughly where the plasma boundary is?" They should answer both correctly without help |

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

### Acceptance Gate

| ID | Type | Criterion | Verification |
|----|------|-----------|--------------|
| 6b.1 | `[AUTO]` | Solov'ev analytic benchmark passes | `tests/test_validation.py::test_solovev_benchmark` — Reconstruct from exact Solov'ev sensor values, assert `chi_squared < 1e-6` and boundary error < 0.1 mm |
| 6b.2 | `[AUTO]` | Noise sweep covers required SNR levels | `tests/test_validation.py::test_noise_sweep_coverage` — Assert reconstruction was run at SNR = ∞, 40, 20, 10 dB and results dict has entries for each |
| 6b.3 | `[AUTO]` | Noise sweep: error increases monotonically with noise | `tests/test_validation.py::test_noise_sweep_monotonic` — Assert Ip error at SNR=10dB > error at SNR=20dB > error at SNR=40dB > error at SNR=∞ |
| 6b.4 | `[AUTO]` | Sensor dropout: 50% removal still converges | `tests/test_validation.py::test_sensor_dropout_50pct` — Remove 50% of sensors, assert reconstruction still converges (doesn't raise) and Ip error < 20% |
| 6b.5 | `[AUTO]` | Sensor dropout: error increases with removal fraction | `tests/test_validation.py::test_dropout_error_increases` — Mean Ip error at 50% removal > mean error at 25% > mean error at 10% |
| 6b.6 | `[AUTO]` | Mesh convergence: outputs converge | `tests/test_validation.py::test_mesh_convergence` — Run at 4 resolutions (N, 2N, 4N, 8N elements). Assert `abs(Ip[8N] - Ip[4N]) < abs(Ip[2N] - Ip[N])` (convergence trend) |
| 6b.7 | `[SCRIPT]` | Validation notebook runs end-to-end | `jupyter nbconvert --to notebook --execute notebooks/validation_report.ipynb --ExecutePreprocessor.timeout=600 --output /dev/null` exits 0 |
| 6b.8 | `[SCRIPT]` | Validation notebook produces all required figures | `jupyter nbconvert --to script notebooks/validation_report.ipynb --stdout \| grep -c "fig.show\|plt.show\|display("` returns >= 4 (at minimum: noise sweep plot, dropout plot, convergence plot, summary table) |
| 6b.9 | `[HUMAN]` | Failure modes are documented | Open `notebooks/validation_report.ipynb` — does it contain a section explicitly titled "Failure Modes" or "Known Limitations" with at least 2 documented failure modes? |

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

### Acceptance Gate

| ID | Type | Criterion | Verification |
|----|------|-----------|--------------|
| 7.1 | `[SCRIPT]` | README exists and has required sections | `grep -q "## Quickstart" README.md && grep -q "## Architecture" README.md && grep -q "spec.md" README.md` — all exit 0 |
| 7.2 | `[SCRIPT]` | Architecture doc exists | `test -f docs/architecture.md && grep -q "data flow" docs/architecture.md` exits 0 |
| 7.3 | `[SCRIPT]` | Operator guide exists and covers required topics | `test -f docs/operator_guide.md && grep -q "new shot" docs/operator_guide.md && grep -q "dashboard" docs/operator_guide.md && grep -q "new sensor" docs/operator_guide.md && grep -q "troubleshoot" docs/operator_guide.md` — all exit 0 |
| 7.4 | `[SCRIPT]` | Demo file exists | `test -f docs/demo.gif -o -f docs/demo.mp4 -o -f docs/demo.webm` exits 0 |
| 7.5 | `[SCRIPT]` | All notebooks execute without error | `for nb in notebooks/*.ipynb; do jupyter nbconvert --to notebook --execute "$nb" --ExecutePreprocessor.timeout=600 --output /dev/null || exit 1; done` exits 0 |
| 7.6 | `[SCRIPT]` | Full CI passes | `gh run list --limit 1 --json conclusion -q '.[0].conclusion'` returns `"success"` |
| 7.7 | `[SCRIPT]` | Release tag exists | `git tag --list 'v0.1.0'` returns `v0.1.0` |
| 7.8 | `[SCRIPT]` | Quickstart works from clean install | `cd /tmp && git clone <repo> && cd cute-pipeline && pip install -e ".[dev]" && python -m src.forward.generate_synthetic && cute-reconstruct --shot data/synthetic/shot_001.h5 --output /tmp/qs_test.h5 && python -m src.dashboard.app --port 8098 &; sleep 3; curl -s -o /dev/null -w '%{http_code}' http://localhost:8098` — each step exits 0, final curl returns 200 |
| 7.9 | `[HUMAN]` | A new student succeeds in < 30 minutes | Find someone who has never seen this repo. Give them only the README. Time them from `git clone` to seeing a reconstructed equilibrium on the dashboard. Did they complete it in under 30 minutes without asking you for help? |

---

## Acceptance Summary

Quick reference: total criteria per phase and how many are automated.

| Phase | Total Criteria | `[AUTO]` | `[SCRIPT]` | `[HUMAN]` |
|-------|---------------|----------|------------|-----------|
| 1. Bootstrap | 9 | 0 | 9 | 0 |
| 2. TokaMaker | 7 | 3 | 3 | 1 |
| 3a. Forward Model | 9 | 5 | 3 | 1 |
| 3b. Signal Processing | 9 | 7 | 1 | 1 |
| 3c. Data Storage | 9 | 7 | 1 | 0 (* see note) |
| 4. OFT Contributions | 3 | 0 | 2 | 1 |
| 5. Reconstruction | 13 | 10 | 2 | 1 |
| 6a. Dashboard | 9 | 5 | 2 | 2 (* 6a.9 counts as 1) |
| 6b. Validation | 9 | 6 | 2 | 1 |
| 7. Documentation | 9 | 0 | 8 | 1 |
| **Total** | **86** | **43** | **33** | **8** (* 2 need external person) |

**50% of all criteria are fully automated pytest tests. 88% require no human judgment.**

---

## CI Integration

All `[AUTO]` tests should be in the pytest suite and run on every push. Suggested test file layout:

```
tests/
├── test_phase2.py          # Phase 2 acceptance tests
├── test_forward.py         # Phase 3a acceptance tests
├── test_signal.py          # Phase 3b acceptance tests
├── test_store.py           # Phase 3c acceptance tests
├── test_reconstruct.py     # Phase 5 acceptance tests
├── test_dashboard.py       # Phase 6a acceptance tests
├── test_validation.py      # Phase 6b acceptance tests
└── conftest.py             # Shared fixtures (reference equilibrium, sensor config, etc.)
```

All `[SCRIPT]` checks can be collected into a single `scripts/acceptance_check.sh` that runs each command and reports pass/fail:

```bash
#!/bin/bash
set -e
PASS=0; FAIL=0

check() {
    if eval "$2" > /dev/null 2>&1; then
        echo "✓ $1"; ((PASS++))
    else
        echo "✗ $1"; ((FAIL++))
    fi
}

check "1.1 OFT importable" "python -c 'import OpenFUSIONToolkit'"
check "1.2 TokaMaker importable" "python -c 'from OpenFUSIONToolkit.TokaMaker import TokaMaker'"
# ... all other [SCRIPT] checks ...

echo ""
echo "Passed: $PASS  Failed: $FAIL"
[ $FAIL -eq 0 ] || exit 1
```

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
