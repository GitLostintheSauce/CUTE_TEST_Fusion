# CLAUDE.md

Working notes for Claude Code in this repository. Read this before changing
anything.

## What this project is

A magnetic equilibrium reconstruction pipeline for CUTE (Columbia University
Tokamak for Education), built on the Open Fusion Toolkit's TokaMaker
Grad-Shafranov solver, plus a from-scratch NumPy neural surrogate that
reconstructs plasma parameters far faster than iterative inversion.

The author is applying to join the CUTE program. **The repo must not claim
affiliation with it.**

## Hard rules

1. **Honesty is the core constraint.** Every physics or ML claim carries a
   label and a validated number. Synthetic data is labeled synthetic
   everywhere, including the dashboard header badge. The reduced forward model
   is labeled reduced. Never let this drift into marketing language. If a
   result is weaker than it looks, say so in the artifact itself.
2. **No em dashes anywhere.** Code, comments, UI strings, docs, commit
   messages. Standing preference.
3. **Never claim affiliation with the CUTE program.**
4. **Do not commit `HANDOFF.md` or `docs/visit_script.md`.** Both are
   gitignored local notes. Personal or career context does not go in tracked
   files.
5. **`.claude/` stays untracked.**
6. **The author has to explain this work in person to fusion physicists.** Code
   she cannot narrate is a liability, not an asset. So: explain the mechanism
   and the tradeoff, not just the diff; define domain jargon the first time it
   appears; state each piece's honest limits so they can be volunteered before
   being challenged; and check understanding by posing the question an expert
   would ask rather than assuming silence means it landed. Teaching beats speed
   when the two conflict.

## Commands

```bash
source .venv/bin/activate

# Dashboard (no hot reload: restart after code changes)
python -m src.dashboard.app --port 8050

# Containerized. Needs a running Docker daemon; this machine uses Colima,
# which does not auto-start, so run `colima start` first.
docker compose up

# Regenerate artifacts
python scripts/generate_synthetic_shot.py     # shots the dashboard reads
python scripts/train_surrogate.py --samples 8000 --epochs 400
python scripts/validate_surrogate.py          # robustness study
python scripts/uncertainty_report.py          # calibration study

# Quality gates, all must pass before commit
ruff check src tests scripts wsgi.py
mypy src/
pytest tests/ -q
```

Full local suite is ~3 minutes because the solver tests actually run.

## Architecture

| Path | What it is |
|---|---|
| `src/reconstruct/` | The only code that touches OFT/TokaMaker |
| `src/forward/` | Sensor geometry, synthetic diagnostics from a solved equilibrium |
| `src/ml/` | Surrogate: reduced forward model, MLP, baseline, validation, uncertainty |
| `src/dashboard/` | Plotly Dash app. Contains no OFT code |
| `src/store/`, `src/signal/` | HDF5 I/O and signal conditioning |
| `scripts/` | Regenerate shots, models, and both reports |
| `models/` | Trained surrogate, robust variant, ensemble, metrics JSON |

Two facts that are easy to get wrong:

- **OFT never runs in the dashboard.** Shots are pre-generated; the dashboard
  reads stored HDF5. `scripts/generate_synthetic_shot.py` is analytic and does
  not import OFT either.
- **OFT never touches the ML surrogate.** The surrogate trains on
  `src/ml/dataset.py`, the reduced analytic forward model. The OFT half and the
  ML half of the repo never call each other.

## Decisions not to undo

1. **The surrogate predicts Ip, R0, Z0, a and deliberately not q95, beta_pol,
   li.** In the current synthetic data those three are computed from Ip by
   formula, so predicting them would be circular. Roadmap 3.2 unblocks them
   once real flux maps exist.
2. **The benchmark is a least-squares inversion of the same reduced model,**
   not a full Grad-Shafranov solve. Say so wherever the speedup appears.
3. **The coverage badge reports what CI verifies (52%), not the nicer local
   77%.** A badge CI cannot back up costs more credibility than the higher
   number buys.
4. **Equilibrium flux contours are labeled "illustrative"** because stored
   equilibria hold only the boundary and scalars, not a psi map. Roadmap 3.1
   fixes this properly.
5. **Prediction and error bar come from the same estimator.** A bug once had
   the point estimate from the single network and the bar from the ensemble, so
   the bar described a different model. Keep them together.
6. **The ML surrogate panel is independent of the shot selection.** It draws
   plasmas from the reduced forward model. Two modes: a new plasma per click,
   or the same plasma with a fresh noise draw, which separates measurement
   error from model error.

## Gotchas

- **OFT is not importable in CI**, so 48 solver tests skip there (54 run).
  The install step downloads it and passes, but the Python import fails on the
  runner. Tracked as roadmap 1.9. This is the whole 52% vs 77% coverage gap.
- **The dashboard dev server does not hot-reload.** Restart it after edits or
  you will be looking at stale code.
- **Sensor noise is currently hardcoded** at `noise_frac=0.02` in the dashboard
  callbacks and is not surfaced in the UI.
- **Needs the author's own hands:** obtaining real CUTE or experimental data,
  anything requiring her hosting or GitHub account, and recording demos of her
  own screen.
