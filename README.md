# CUTE Tokamak Magnetic Diagnostic Pipeline

A complete magnetic equilibrium reconstruction pipeline for Columbia University's CUTE (Columbia University Tokamak for Education) spherical torus. Built on the [Open Fusion Toolkit (OFT)](https://github.com/hansec/OpenFUSIONToolkit) TokaMaker Grad-Shafranov solver, this project processes synthetic diagnostic signals, reconstructs plasma equilibria, and provides an interactive dashboard for shot review.

## Quickstart

```bash
# Clone and install
git clone https://github.com/GitLostintheSauce/CUTE_TEST.git
cd CUTE_TEST
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Install OFT (see docs/operator_guide.md for details)
source scripts/setup_env.sh

# Run tests
pytest tests/ -v

# Launch the dashboard
python -m src.dashboard.app --port 8050
```

## Architecture

```
Raw Signals ──► Signal Processing ──► Reconstruction ──► Dashboard
  (HDF5)     (bandpass, notch,      (iterative GS     (Plotly Dash
              drift correction)      solve via OFT)    web app)

┌─────────────────────────────────────────────────────────┐
│                    src/ modules                          │
│                                                         │
│  store/         Signal & shot data I/O (HDF5 + Pydantic)│
│  signal/        Bandpass, notch, integration, calibration│
│  forward/       Sensor geometry + synthetic diagnostics  │
│  reconstruct/   Iterative equilibrium reconstruction     │
│  validation/    Noise sweep, dropout, convergence tests  │
│  dashboard/     Plotly Dash interactive web app          │
└─────────────────────────────────────────────────────────┘
```

**Data flow:** raw HDF5 → `signal.processing.process_pipeline()` → `reconstruct.fit_equilibrium()` → `store.hdf5.save_shot()` → `dashboard.app`

See [spec.md](spec.md) for full project specification, phase DAG, and acceptance criteria.

## Architecture Details

See [docs/architecture.md](docs/architecture.md) for module responsibilities, design decisions, and data flow diagrams.

## Operator Guide

See [docs/operator_guide.md](docs/operator_guide.md) for how to process shots, use the dashboard, add new sensors, and troubleshoot common issues.

## Project Structure

```
CUTE_TEST/
├── src/
│   ├── store/          # Pydantic schemas + HDF5 I/O
│   ├── signal/         # Signal processing pipeline
│   ├── forward/        # Sensor config + forward model
│   ├── reconstruct/    # Equilibrium reconstruction
│   ├── validation/     # Benchmarks and validation
│   └── dashboard/      # Plotly Dash web app
├── tests/              # pytest test suite (50 tests)
├── data/               # CUTE mesh + shot data
├── config/             # Processing parameters (TOML)
├── notebooks/          # Jupyter notebooks
├── docs/               # Documentation
└── spec.md             # Full project specification
```

## Key Dependencies

- **OpenFUSIONToolkit** — Fortran-backed FEM plasma simulation (TokaMaker GS solver)
- **NumPy/SciPy** — Signal processing and numerical computation
- **Pydantic v2** — Data validation schemas
- **HDF5 (h5py)** — Shot data storage
- **Plotly Dash** — Interactive dashboard
- **pytest** — Test framework

## License

This project is part of the CUTE tokamak educational program at Columbia University.
