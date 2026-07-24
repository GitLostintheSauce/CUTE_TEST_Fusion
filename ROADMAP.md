# CUTE Pipeline : Magnum Opus Roadmap

A staged plan to grow this project from a strong diagnostic pipeline into a
portfolio centerpiece that reads convincingly to **scientific-software**,
**ML/data-science**, and **general-SWE/backend** reviewers at once.

Every idea from the brainstorm is captured here, each with an **Awesomeness
rating** (portfolio impact) and an **Effort** estimate. Nothing is deleted;
things move from "Planned" to "Done" as we go. We build step by step and keep
each stage polished before moving on.

Guiding principle: **credibility is the asset.** Every physics or ML claim is
labeled for what it is and validated with a number. "ML-accelerated
reconstruction, benchmarked against a solver with reported error" is a great
story; "AI-powered plasma prediction" with no validation is not.

Awesomeness scale: ★ (nice) → ★★★★★ (interviewer leans in).

---

## Stage 0 : Where we are (Done)

- Full pipeline: HDF5 store, signal processing, forward model, EFIT-style
  reconstruction, validation, Plotly Dash dashboard.
- Professional dashboard: sticky header, sidebar, responsive plot grid,
  colorblind-safe palette, honest "synthetic demo data" labeling, real flux
  contour visualization, About panel.
- Passing test suite.

---

## Stage 1 : The multipliers (cheap, enormous payoff)

The "why haven't I done this yet" wins. A reviewer spends ~90 seconds; these
decide whether they spend more.

| # | Idea | Awesome | Effort |
|---|------|---------|--------|
| 1.1 | **Deploy live** : public URL (Fly.io / Render / HF Spaces) with a Dockerfile so anyone can click and see it work | ★★★★★ | S |
| 1.2 | **GitHub Actions CI** : tests + `ruff` + `mypy` on every push, with badges | ★★★★ | S |
| 1.3 | **Portfolio README** : hero GIF, architecture diagram, results section, one-paragraph pitch | ★★★★ | S |
| 1.4 | **Dockerfile + docker-compose** : `docker compose up` Just Works | ★★★★ | S |
| 1.5 | **Coverage + pre-commit hooks** : coverage badge, auto-format/lint on commit | ★★★ | S |

---

## Stage 2 : The standout technical story (our differentiator)

This is what turns "nice dashboard" into "wait, tell me about this."

| # | Idea | Awesome | Effort |
|---|------|---------|--------|
| 2.1 | **ML surrogate for equilibrium reconstruction** : NN maps sensor signals → plasma parameters ~100-1000× faster than the iterative solve, with a quantified accuracy + speed benchmark. Active research area (DeepMind, EPFL, PPPL). | ★★★★★ | M |
| 2.2 | **Reduced-physics forward model** : analytic circular-loop Green's functions (elliptic integrals) to generate a large labeled dataset without needing the heavy TokaMaker install; validated against Biot-Savart quadrature | ★★★★ | M |
| 2.3 | **Close the physics loop honestly** : persist the full ψ (flux) map from TokaMaker and render the real thing; deletes the "illustrative" caveat and enables a true forward-model "measured vs. simulated" comparison | ★★★★ | M |
| 2.4 | **Validation / benchmark report** : reconstruction accuracy vs. ground truth, noise-robustness sweeps, sensor-dropout studies ("lose 20% of Mirnov probes, still within X%") | ★★★★ | M |
| 2.5 | **Uncertainty quantification** : error bars on reconstructed parameters (ensemble or MC-dropout). Almost nobody does this; signals scientific maturity | ★★★★ | M |
| 2.6 | **Disruption / anomaly detection** : classifier that flags disruptive or off-normal shots from signal features | ★★★★ | M |

---

## Stage 3 : The systems-engineer flex

| # | Idea | Awesome | Effort |
|---|------|---------|--------|
| 3.1 | **FastAPI service layer** : reconstruction exposed as a REST API with auto OpenAPI docs; the dashboard becomes one client of a clean backend | ★★★★ | M |
| 3.2 | **Real-time streaming mode** : simulated data-acquisition system streaming samples over WebSocket, dashboard updating live like a control room (REC indicator, rolling window) | ★★★★★ | M |
| 3.3 | **Job/queue architecture** : submit reconstruction jobs, poll status; shows async/worker design | ★★★ | M |

---

## Stage 4 : The moonshots (do NOT do all of these)

The top shelf. Any one of these is a career-long talking point.

| # | Idea | Awesome | Effort |
|---|------|---------|--------|
| 4.1 | **Closed-loop plasma control with RL** : an agent learns to drive coil currents to a target plasma shape. This is the DeepMind *Nature* 2022 TCV paper, scaled to CUTE. Even a toy version is extraordinary | ★★★★★ | XL |
| 4.2 | **Interactive "what-if" equilibrium** : sliders for coil currents; drag them and watch the plasma boundary respond live via the forward model | ★★★★★ | L |
| 4.3 | **Real open experimental data** : bring in a public tokamak dataset for at least one panel; kills the "it's all synthetic" risk entirely | ★★★★★ | L |
| 4.4 | **Physics-informed neural network (PINN)** : a network that solves the Grad-Shafranov PDE directly | ★★★★ | XL |

---

## Recommended walking order

1. **Stage 2.1 + 2.2 (ML surrogate) : DONE.** The technical story first,
   because it's the hook everything else hangs on. See `src/ml/`,
   `scripts/train_surrogate.py`, `models/surrogate_metrics.json`, and the
   live panel in the dashboard. Held-out R2 = 0.99, ~13,000x faster than the
   iterative baseline. Next: Stage 2.4 validation report, then Stage 1.
2. Stage 1 (deploy + CI + README) : package it so people can see it.
3. Stage 2.4 (validation report) + Stage 3.2 (streaming) : depth and dazzle.
4. Pick one Stage 4 moonshot to chase.

## Honesty guardrails (permanent)

- Synthetic data stays labeled as synthetic.
- The reduced forward model is labeled as reduced (not full free-boundary GS).
- ML claims ship with a validation number and a stated baseline.
- No invented physical constants; sensor geometry and formulas are cited in code.
