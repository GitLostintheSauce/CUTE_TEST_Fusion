# CUTE Pipeline: Magnum Opus Roadmap

A staged plan to grow this project from a strong diagnostic pipeline into a
portfolio centerpiece that reads convincingly to **scientific-software**,
**ML/data-science**, and **general-SWE/backend** reviewers at once.

This document is ordered **sequentially by dependency**, not by importance.
Each phase assumes the previous one is finished. Every idea from the original
brainstorm is preserved, with an **Awesomeness rating** (portfolio impact) and
an **Effort** estimate. Nothing is deleted; items move from TODO to DONE.

Awesomeness scale: 1 star (nice) to 5 stars (interviewer leans in).
Effort: S (under an hour), M (a day), L (a few days), XL (a project).

Guiding principle: **credibility is the asset.** Every physics or ML claim is
labeled for what it is and validated with a number. "ML-accelerated
reconstruction, benchmarked against a solver with reported error" is a great
story; "AI-powered plasma prediction" with no validation is not.

---

## Phase 0: Foundation (DONE)

- Full pipeline: HDF5 store, signal processing, forward model, EFIT-style
  reconstruction, validation, Plotly Dash dashboard.
- Professional dashboard: sticky header, sidebar, responsive plot grid,
  colorblind-safe palette, honest "synthetic demo data" labeling, real flux
  contour visualization, About panel.
- Passing test suite.

## Phase 0.5: The technical story (DONE, out of sequence)

Built early because it is the hook everything else hangs on.

| # | Item | Awesome | Effort | Status |
|---|------|---------|--------|--------|
| A | **ML surrogate for equilibrium reconstruction.** Neural net maps 130 sensor signals to plasma parameters, far faster than iterative reconstruction, with a quantified accuracy and speed benchmark. Active research area (DeepMind, EPFL, PPPL). | 5 | M | DONE |
| B | **Reduced-physics forward model.** Analytic circular-loop Green's functions (elliptic integrals) generate a large labeled dataset without the heavy TokaMaker install; validated to machine precision against Biot-Savart quadrature. | 4 | M | DONE |

Result: held-out R2 = 0.99, roughly 13,000x faster than the iterative
baseline. See `src/ml/`, `scripts/train_surrogate.py`,
`models/surrogate_metrics.json`, and the live dashboard panel.

**Known deviation from the original pitch, to be closed in Phase 3:** the
surrogate predicts Ip, R0, Z0, and a (current, major radius, vertical
position, minor radius), **not** q95, beta_pol, and li. In the current
synthetic data those three are computed by formula directly from Ip, so
predicting them from signals would be circular and meaningless. They become
genuinely predictable once real flux maps or real experimental data land.
Also note the benchmark is against a least-squares inversion of the same
reduced model, not against the full Grad-Shafranov solve.

## Phase 1: Make the repo look like a company owns it (DONE)

Green badges at the top of a README are a competence signal people read
subconsciously. A reviewer spends about 90 seconds; this phase decides whether
they spend more.

| # | Item | Awesome | Effort | Status |
|---|------|---------|--------|--------|
| 1.1 | **GitHub Actions CI**: tests + lint on every push | 4 | S | DONE |
| 1.2 | **ruff + mypy clean** across the codebase | 4 | S | DONE |
| 1.3 | **pre-commit hooks**: auto-format and lint on commit | 3 | S | DONE |
| 1.4 | **Dockerfile + docker-compose** so `docker compose up` Just Works | 4 | S | DONE |
| 1.5 | **README badges**: CI status, Python version | 4 | S | DONE |
| 1.6 | **Coverage badge** (52% CI-verified, enforced by --cov-fail-under=50; 77% locally with OFT) | 3 | S | DONE |
| 1.7 | **CI triggers on every push**, not only pushes to main and PRs to main | 3 | S | DONE |
| 1.8 | **Deployment config**: `wsgi.py` gunicorn entrypoint, `render.yaml` blueprint | 5 | S | DONE |
| 1.9 | **Make OFT importable in CI.** The install step downloads the toolkit and passes, but the Python import still fails on the runner, so 48 solver-dependent tests skip (44 run). That is why CI-verified coverage is 52% rather than the 77% seen locally. Fixing this would raise verified coverage substantially. | 3 | M | TODO |

## Phase 2: Housekeeping (do before anything else new)

Zero dependencies, so every later commit sits on a clean base.

| # | Item | Awesome | Effort | Status |
|---|------|---------|--------|--------|
| 2.1 | **Restore `spec.md`**. It currently has a stray `222` on line 1 and about 260 deleted lines from an accidental edit. | 2 | S | DONE |
| 2.2 | **Add a LICENSE file** and restore the matching README badge. Settles what the README is allowed to claim. | 3 | S | TODO |

## Phase 3: Settle the physics foundation

Change the foundation before writing reports about it, or the reports go stale.

| # | Item | Awesome | Effort | Status |
|---|------|---------|--------|--------|
| 3.1 | **Close the physics loop honestly.** Persist the full psi (flux) map from TokaMaker and render the real thing. Deletes the "illustrative" caveat on the equilibrium view and enables a true forward-model "measured vs. simulated" comparison. | 4 | M | TODO |
| 3.2 | **Extend the surrogate to q95, beta_pol, li** once psi maps make them non-circular. Closes the Phase 0.5 deviation. | 4 | M | TODO |

## Phase 4: Measure what you built

| # | Item | Awesome | Effort | Status |
|---|------|---------|--------|--------|
| 4.1 | **Validation / benchmark report.** Reconstruction accuracy vs. ground truth, noise-robustness sweeps, sensor-dropout studies ("lose 20% of Mirnov probes, still within X%"). | 4 | M | DONE |
| 4.2 | **Uncertainty quantification.** Error bars on reconstructed parameters (ensemble or MC-dropout). Almost nobody does this; signals scientific maturity. | 4 | M | TODO |

Order matters here: you cannot sensibly model uncertainty until you have
characterized how the error actually behaves, and 4.1 produces exactly that.

## Phase 5: Additional ML feature

| # | Item | Awesome | Effort | Status |
|---|------|---------|--------|--------|
| 5.1 | **Disruption / anomaly detection.** Classifier that flags disruptive or off-normal shots from signal features. | 4 | M | TODO |

Slotted after the core ML is validated so you are never debugging two learning
systems at once, and so it can reuse the Phase 4 validation tooling.

## Phase 6: Restructure the backend

Refactoring while features churn is wasted work, so this comes after the
feature set is stable. Within the phase, the API comes first because streaming
and the job queue both hang off it.

| # | Item | Awesome | Effort | Status |
|---|------|---------|--------|--------|
| 6.1 | **FastAPI service layer.** Reconstruction exposed as a REST API with auto-generated OpenAPI docs; the dashboard becomes one client of a clean backend. | 4 | M | TODO |
| 6.2 | **Real-time streaming mode.** Simulated data-acquisition system streaming samples over WebSocket, dashboard updating live like a control room (REC indicator, rolling window). | 5 | M | TODO |
| 6.3 | **Job / queue architecture.** Submit reconstruction jobs, poll status; shows async and worker design. | 3 | M | TODO |

## Phase 7: Pick AT MOST ONE moonshot

The top shelf. Any one of these is a career-long talking point. Doing all of
them is how the project never ships.

| # | Item | Awesome | Effort | Status |
|---|------|---------|--------|--------|
| 7.1 | **Interactive "what-if" equilibrium.** Sliders for coil currents; drag them and watch the plasma boundary respond live via the forward model. Most demo value per unit effort. Depends on Phase 3 and benefits from Phase 6. | 5 | L | TODO |
| 7.2 | **Real open experimental data.** Bring in a public tokamak dataset for at least one panel. Moving beyond synthetic data, even for one panel, is a huge credibility jump and kills the "it's all made up" risk entirely. Must land before Phase 8 because it changes what the site claims. | 5 | L | TODO |
| 7.3 | **Closed-loop plasma control with reinforcement learning.** An RL agent learns to drive the coil currents to achieve a target plasma shape. This is the DeepMind *Nature* 2022 paper on TCV, scaled down to CUTE. Even a toy version is a talking point for the rest of your career. | 5 | XL | TODO |
| 7.4 | **Physics-informed neural network (PINN).** A network that solves the Grad-Shafranov PDE directly. | 4 | XL | TODO |

## Phase 8: Final polish (only once the UI is frozen)

All three describe the finished product, so doing them earlier guarantees
rework. This is why the demo GIF is here and not at the start.

| # | Item | Awesome | Effort | Status |
|---|------|---------|--------|--------|
| 8.1 | **Demo GIF and fresh screenshots** for the README. Recruiters and busy professors skim; a moving image is what they stop on. | 4 | S | TODO |
| 8.2 | **Architecture diagram** and a final README pass. | 3 | S | TODO |
| 8.3 | **Honesty-label audit.** Re-check that every caveat still matches what the code actually does. | 4 | S | TODO |

## Phase 9: Ship

| # | Item | Awesome | Effort | Status |
|---|------|---------|--------|--------|
| 9.1 | **Pre-flight.** CI green, Docker image builds, surrogate model and demo data present in the image. | 4 | S | TODO |
| 9.2 | **Deploy live.** Public URL via Render, Fly.io, or Hugging Face Spaces so anyone can click and see it work. Deliberately the final step. | 5 | S | TODO |

---

## Notes

- You can stop after any phase and still have a coherent, strong project.
  Phases 1 through 4 alone leave it in excellent shape.
- Keeping tests and CI green is a continuous habit, not a numbered step.
- The biggest risk is not "it isn't good enough." It is polishing forever and
  never shipping.

## Honesty guardrails (permanent)

- Synthetic data stays labeled as synthetic.
- The reduced forward model is labeled as reduced, not full free-boundary GS.
- ML claims ship with a validation number and a stated baseline.
- No invented physical constants; sensor geometry and formulas are cited in code.
- Benchmarks state what they are measured against.
