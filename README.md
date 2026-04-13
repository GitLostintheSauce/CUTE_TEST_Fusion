# CUTE Tokamak Magnetic Diagnostic Pipeline

Python package for processing magnetic diagnostics from Columbia University's CUTE spherical tokamak, reconstructing equilibria with OFT/TokaMaker, and reviewing results in a Dash dashboard.

This repository currently includes:
- Signal processing utilities (`src/signal`)
- HDF5 shot storage + schemas (`src/store`)
- Synthetic forward-model helpers (`src/forward`)
- Constraint and EFIT-style reconstruction code, including eddy-current compensation (`src/reconstruct`)
- Validation/benchmark tooling (`src/validation`)
- A Dash app for shot browsing and plotting (`src/dashboard`)

## Install

```bash
git clone https://github.com/GitLostintheSauce/CUTE_TEST.git
cd CUTE_TEST
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## External dependency for reconstruction

The reconstruction modules use [OpenFUSIONToolkit (OFT)](https://github.com/hansec/OpenFUSIONToolkit) / TokaMaker. If OFT is not installed, reconstruction CLI commands will fail with an import error, while pure signal/store utilities and many tests still work.

For environment setup details, see `scripts/setup_env.sh` and `docs/operator_guide.md`.

## Quick usage

### 1) Run tests

```bash
pytest tests/ -v
```

### 2) Run reconstruction CLI

```bash
cute-reconstruct --shot data/synthetic/shot_001.h5 --output /tmp/results.h5
# optional: --method efit|constraint
```

### 3) Launch dashboard

```bash
python -m src.dashboard.app --port 8050
# optional auth gate: --token mysecrettoken
```

## Current data flow in code

```text
HDF5 shot data
  -> signal processing (filtering/integration/calibration)
  -> optional eddy-current compensation
  -> reconstruction (constraint or EFIT-style)
  -> saved shot/equilibrium outputs
  -> dashboard visualization
```

## Project layout

```text
CUTE_TEST/
├── src/
│   ├── store/        # HDF5 I/O + data schemas
│   ├── signal/       # Filtering, integration, calibration pipeline
│   ├── forward/      # Sensor geometry + synthetic diagnostic model
│   ├── reconstruct/  # Constraint solver, EFIT, Green's, eddy compensation, CLI
│   ├── validation/   # Benchmarks and sensor placement utilities
│   └── dashboard/    # Plotly Dash app
├── tests/            # pytest suite
├── docs/             # Architecture, operator guide, eddy-current notes
├── config/           # Processing configuration
└── scripts/          # Environment setup + synthetic-shot helpers
```

## Key commands

```bash
# formatter/lint/type checks (if installed)
ruff check .
mypy src

# run one focused integration test
pytest tests/test_integration.py -v
```

## Notes on current implementation

- The dashboard includes shot browsing, signal plots, equilibrium contours, parameter timelines, and a sim-vs-experiment panel.
- The sim-vs-experiment synthetic trace in the dashboard is currently a simple baseline derived from measured data (demo behavior), not a full dynamic forward-model replay.
- `cute-reconstruct` is exposed as a console script from `pyproject.toml`.

## Documentation

- `docs/architecture.md` — module responsibilities and design choices
- `docs/operator_guide.md` — workflows for processing, reconstruction, dashboard use, and troubleshooting
- `docs/eddy_currents.md` — eddy-current model and compensation details
- `spec.md` — full project plan/specification and acceptance criteria

## Challenge-winning tips (for your "$100 challenge")

1. **Optimize for judged criteria, not effort.** Make a checklist from the rubric and map every deliverable to evidence (test output, screenshot, short demo clip).
2. **Demo reliability beats extra features.** A smaller feature set that never breaks will score better than ambitious unstable work.
3. **Show before/after metrics.** Include 2–3 concrete numbers (runtime, error, pass rate) so judges can see improvement quickly.
4. **Tell a tight story.** Problem → approach → result in under 90 seconds. Most people lose points in explanation, not coding.
5. **Pre-record a fallback demo.** If live demo fails, have a short backup clip ready.
6. **Ship polish fast.** Clear README, one-command setup, and obvious commands (`make test`, `make demo`) can be the difference-maker.

You got this — play smart and make it easy for judges to say "yes."

## License

Part of the CUTE tokamak educational pipeline work at Columbia University.
