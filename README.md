# CUTE Tokamak Magnetic Diagnostic Pipeline

![CI](https://github.com/GitLostintheSauce/CUTE_TEST_Fusion/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-52%25%20(CI)-yellow.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

A complete magnetic equilibrium reconstruction pipeline for Columbia University's CUTE (Columbia University Tokamak for Education) spherical torus. Built on the [Open Fusion Toolkit (OFT)](https://github.com/hansec/OpenFUSIONToolkit) TokaMaker Grad-Shafranov solver, this project processes diagnostic signals, reconstructs plasma equilibria, and provides an interactive dashboard for shot review.

**In one sentence:** a tokamak cannot be measured directly, so this pipeline reconstructs the invisible plasma from 130 external magnetic sensors, and includes a from-scratch neural-network surrogate that does that reconstruction roughly 13,000x faster than the classical iterative method.

> Note: the shots shown are synthetic (pipeline-generated) test data. No experimental CUTE data is included, and every panel is labeled accordingly.

> Coverage note: the badge reports what CI verifies (52%), where the 48
> solver-dependent tests skip because the Open Fusion Toolkit is not currently
> importable on the runner. With OFT installed locally the full suite runs
> (86 tests) and coverage reaches 77%. Getting OFT working in CI is a tracked
> TODO in [ROADMAP.md](ROADMAP.md).

![ML surrogate vs. iterative benchmark](docs/surrogate_benchmark.png)

## Run it in one command (Docker)

No Python or solver setup needed:

```bash
docker compose up
# then open http://localhost:8050
```

## Quickstart (local dev)

```bash
# Clone and install
git clone https://github.com/GitLostintheSauce/CUTE_TEST_Fusion.git
cd CUTE_TEST_Fusion

# Requires Python 3.10+ (3.12 recommended). Note: the python3 that ships
# with macOS is 3.9 and its bundled pip is too old for this install; use a
# python.org or Homebrew Python instead (e.g. python3.12 below).
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Generate the synthetic demo shot (used by the dashboard)
python scripts/generate_synthetic_shot.py

# (Optional) train the ML surrogate
python scripts/train_surrogate.py --samples 8000 --epochs 400

# Run tests (solver-dependent tests need OFT; see docs/operator_guide.md)
pytest tests/ -v

# Launch the dashboard
python -m src.dashboard.app --port 8050
```

## Deploy a public link

The included `Dockerfile` and `render.yaml` make deployment one step:

- **Render.com:** New + -> Blueprint -> point at this repo. Render reads `render.yaml`, builds the Docker image, and gives you a public URL.
- **Hugging Face Spaces:** create a Docker Space and push this repo; it builds the same `Dockerfile`.

The dashboard runs without the OFT solver (viewing shots and the ML surrogate do not need it), which keeps the deployed image small.

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

**Data flow:** raw HDF5 → `signal.processing.process_pipeline()` → eddy compensation → EFIT reconstruct → `store.hdf5.save_shot()` → `dashboard.app`

### Advanced Reconstruction (v0.2.0)

- **Green's function matrix**: Pre-computed 130×28 matrix relating coil currents to sensor measurements, enabling fast vacuum field decomposition
- **EFIT-style reconstruction**: Iterative decomposition of measured fields into coil + plasma contributions using Tikhonov-regularized least-squares
- **Eddy current compensation**: Exponential decay model of vacuum vessel eddy currents (3 eigenmodes, τ = 30–105 μs), subtracted from measurements before reconstruction
- **Sensor placement optimization**: Fisher information analysis, greedy forward selection, and leave-one-out importance ranking to identify minimum viable sensor sets

See [spec.md](spec.md) for full project specification, phase DAG, and acceptance criteria.

### ML Surrogate Reconstruction (v0.3.0)

A neural-network surrogate that maps the 130 magnetic diagnostic signals
directly to plasma parameters (plasma current, major radius, vertical
position, minor radius), as a fast alternative to iterative reconstruction.

- **From-scratch NumPy MLP** (`src/ml/mlp.py`): forward pass, backprop, Adam,
  and standardization implemented without a deep-learning framework, so the
  feature adds zero heavy dependencies and stays fully reproducible.
- **Physics-grounded training data** (`src/ml/physics.py`, `dataset.py`): a
  reduced forward model built from analytic circular-loop Green's functions
  (elliptic integrals), validated to machine precision against a direct
  Biot-Savart quadrature and the exact on-axis field.
- **Honest benchmark** (`src/ml/baseline.py`): the surrogate is compared
  against a classical nonlinear least-squares inversion of the *same* physics.

Results on a held-out set (`models/surrogate_metrics.json`):

| Metric | Value |
|--------|-------|
| R² (overall) | 0.99 |
| Plasma current MAE | ~1.8 kA (on a 20-250 kA range) |
| Position accuracy (R0, Z0) | ~1 mm |
| Inference speed | ~1 µs/shot |
| Speedup vs. iterative baseline | ~13,000× |

The iterative baseline is slightly more accurate; the surrogate trades a small
accuracy cost for a large speed gain, which is the tradeoff that makes
real-time reconstruction feasible. Train it with:

```bash
python scripts/train_surrogate.py --samples 8000 --epochs 400
```

![Surrogate vs. iterative benchmark](docs/surrogate_benchmark.png)

The dashboard's **ML Surrogate Reconstruction (live)** panel samples a plasma,
adds measurement noise, and reconstructs it on click, showing predicted vs.
true parameters and the inference latency.

### Robustness validation

Real diagnostics are noisy and probes fail, so the surrogate is stress-tested
rather than only reported at its best. Full numbers in
[docs/validation_report.md](docs/validation_report.md), regenerated by
`python scripts/validate_surrogate.py`.

![Robustness](docs/surrogate_validation.png)

**What the study found:**

1. **Noise tolerance is strong.** Accuracy holds at R2 ~0.90 under 10% sensor
   noise, five times the noise the model was trained on.
2. **Sensor failure was a genuine weakness.** The baseline model degrades
   sharply when channels go dead, because it never saw dead channels in
   training: R2 falls from 0.98 to 0.64 with 20% of probes lost.
3. **Dropout augmentation fixes it.** Retraining with randomly masked input
   channels restores performance under probe failure:

| 20% of sensors dead | Overall R2 | Ip error |
|---|---|---|
| Baseline | 0.64 | 14.0 kA |
| Dropout-augmented | **0.94** | **2.5 kA** |

The tradeoff is honest and documented: the robust model gives up some
clean-signal accuracy (R2 0.99 to 0.95) in exchange for tolerating failure.
Both are shipped, as `models/surrogate.npz` and
`models/surrogate_robust.npz`.

### Uncertainty quantification

An operator needs `Ip = 184 kA +/- 3 kA`, not just `184 kA`. A deep ensemble
of 5 networks provides error bars, and the error bars are then checked rather
than assumed. Full detail in
[docs/uncertainty_report.md](docs/uncertainty_report.md), regenerated by
`python scripts/uncertainty_report.py`.

![Uncertainty](docs/surrogate_uncertainty.png)

**The finding:** raw ensemble spread is *not* calibrated. For honest 1-sigma
bars the truth should land inside about 68% of the time. It did not, and it
missed in both directions:

| Parameter | 1-sigma coverage (raw) | Verdict |
|---|---|---|
| Ip | 0.90 | too wide (over-cautious) |
| Z0 | 0.88 | too wide |
| R0 | 0.76 | slightly wide |
| a | 0.56 | too narrow (overconfident) |

**The fix:** a per-parameter variance rescaling, fitted on a calibration split
that is disjoint from the test split so the result is not circular. Coverage
then lands near nominal (Ip 0.90 to 0.68, `a` 0.56 to 0.72).

Calibrated error bars appear live in the dashboard's surrogate panel, with
errors exceeding 1 sigma highlighted. Note the correlation between predicted
sigma and realized error is positive but modest (0.15 to 0.42), so the
estimate flags which reconstructions are less reliable without being a precise
per-shot error predictor. It is not presented as one.

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
│   ├── reconstruct/    # Equilibrium reconstruction (constraint + EFIT + eddy)
│   ├── validation/     # Benchmarks, validation, sensor placement
│   └── dashboard/      # Plotly Dash web app
├── tests/              # pytest test suite (76 tests)
├── data/               # CUTE mesh + shot data
├── config/             # Processing parameters (TOML)
├── notebooks/          # Jupyter notebooks
├── docs/               # Documentation
└── spec.md             # Full project specification
```

## Key Dependencies

- **OpenFUSIONToolkit**: Fortran-backed FEM plasma simulation (TokaMaker GS solver)
- **NumPy/SciPy**: Signal processing and numerical computation
- **Pydantic v2**: Data validation schemas
- **HDF5 (h5py)**: Shot data storage
- **Plotly Dash**: Interactive dashboard
- **pytest**: Test framework


## License

Released under the [MIT License](LICENSE).
