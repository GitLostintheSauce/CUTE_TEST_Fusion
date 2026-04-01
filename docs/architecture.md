# Architecture

## Data Flow
<!-- data flow diagram -->

```
                    ┌──────────────┐
                    │  CUTE Device │
                    │  (hardware)  │
                    └──────┬───────┘
                           │ analog signals
                           ▼
                    ┌──────────────┐
                    │  Digitizer   │
                    │  → raw HDF5  │
                    └──────┬───────┘
                           │ data/raw/shot_NNN.h5
                           ▼
              ┌────────────────────────┐
              │   Signal Processing    │
              │   src/signal/          │
              │                        │
              │  1. Dropout fix        │
              │  2. Calibration        │
              │  3. Bandpass filter    │
              │  4. 60Hz notch filter  │
              │  5. Integration (dB/dt)│
              └────────┬───────────────┘
                       │ processed signals
                       ▼
              ┌────────────────────────┐
              │  Reconstruction        │
              │  src/reconstruct/      │
              │                        │
              │  1. Estimate Ip        │
              │  2. Set constraints    │
              │  3. TokaMaker solve    │
              │  4. Check convergence  │
              │  5. Extract results    │
              └────────┬───────────────┘
                       │ EquilibriumResult
                       ▼
              ┌────────────────────────┐
              │  Storage               │
              │  src/store/            │
              │                        │
              │  HDF5 with Pydantic    │
              │  validated schemas     │
              └────────┬───────────────┘
                       │
                       ▼
              ┌────────────────────────┐
              │  Dashboard             │
              │  src/dashboard/        │
              │                        │
              │  Plotly Dash web app   │
              │  Shot browser, signal  │
              │  viewer, equilibrium   │
              │  plots, export         │
              └────────────────────────┘
```

## Module Responsibilities

### `src/store/` — Data I/O Layer
- **schemas.py**: Pydantic v2 models (`ShotMetadata`, `SignalMetadata`, `EquilibriumResult`) that define the data contracts
- **hdf5.py**: HDF5 read/write backend. File structure: `/meta`, `/raw/{channel}`, `/processed/{channel}`, `/equilibrium`
- Boundary: everything that touches disk goes through this module

### `src/signal/` — Signal Processing
- **filters.py**: Butterworth bandpass (SOS format for stability at high sample rates) and IIR notch (60Hz + harmonics)
- **integrate.py**: Cumulative trapezoid integration with 3-step drift correction (DC removal → integrate → detrend)
- **calibrate.py**: Sensor calibration (gain + offset)
- **outliers.py**: NaN dropout detection and interpolation
- **processing.py**: Orchestrates the full pipeline: dropout fix → calibrate → bandpass → notch → integrate

### `src/forward/` — Synthetic Diagnostics
- **sensors.py**: CUTE sensor geometry — 56 flux loops + 74 Mirnov probes placed around the vacuum vessel
- **model.py**: Forward model evaluating psi and B at sensor locations via TokaMaker field evaluators
- **noise.py**: Noise models (white, 60Hz pickup, dropout) for synthetic data generation

### `src/reconstruct/` — Equilibrium Reconstruction
- **constraints.py**: Converts sensor measurements to/from ordered numpy vectors, estimates Ip from Mirnov data
- **solver.py**: Iterative reconstruction loop — sets TokaMaker constraints from measurements, solves, checks residual, optionally refines with Jacobian-based coil current adjustment
- **timeseries.py**: Loops over time slices with warm-starting (preserves psi between slices)
- **cli.py**: `cute-reconstruct` command-line entry point

### `src/validation/` — Benchmarks
- **benchmarks.py**: Noise sweep (SNR parameterized), sensor dropout, convergence analysis utilities

### `src/dashboard/` — Web Interface
- **app.py**: Plotly Dash application with shot browser, signal viewer, equilibrium contour plot, parameter timeline, sim-vs-experiment comparison, and token-based auth gate

## Key Design Decisions

### 1. Single TokaMaker Instance
OFT allows only one TokaMaker per process. All code shares a single session-scoped instance. Tests use a `tokamaker_session` pytest fixture. This constraint drove the sequential design of the reconstruction loop.

### 2. SOS Filter Format
Butterworth filters are implemented in second-order sections (SOS) format rather than transfer function (b, a) format. At CUTE's 100kHz sample rate, the standard form has catastrophic numerical instability for even moderate filter orders.

### 3. Pydantic v2 for Data Contracts
All data structures that cross module boundaries use Pydantic models. This provides validation at ingestion time and clean JSON serialization for HDF5 attribute storage.

### 4. Forward Model as Ground Truth
The synthetic data pipeline (forward model + noise) serves as the primary validation path. Zero-noise round-trip tests verify that reconstruction exactly recovers known equilibria.

### 5. Constraint-Based Reconstruction
Rather than directly optimizing coil currents, the reconstruction leverages TokaMaker's built-in constraint solver (isoflux + saddle points + Ip target). This is both faster and more robust than external optimization, as TokaMaker handles the nonlinear GS equation internally.
