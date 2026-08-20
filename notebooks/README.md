# Notebooks

A followable walkthrough of the pipeline, in the style of the Open Fusion
Toolkit's `src/examples/TokaMaker` notebooks. This means that I have the physics motivation in
the markdown, one idea per cell, and caveats stated where they apply rather than
collected in a footnote. This is the first of my workflow notebooks, and I am still working on fine-tuning my dashboard, so I will edit this notebook accordingly.

CUTE is a teaching machine, so the pipeline is more useful as something a class
can reconstruct than as a finished application.

| Notebook | What it covers |
|---|---|
| [00_machine_setup.ipynb](00_machine_setup.ipynb) | The machine: load the mesh, find the coils, solve one equilibrium, and check it four ways. Two tiers of check, machine-agnostic and published-reference. |
| [01_synthetic_diagnostics.ipynb](01_synthetic_diagnostics.ipynb) | The forward map: from a solved equilibrium to what each of the 130 magnetic sensors would read, with measurement noise and sensor loss. |
| [02_reconstruction.ipynb](02_reconstruction.ipynb) | The inverse map: from sensor signals back to plasma parameters by least-squares inversion. Covers the inverse crime, what noise does to the answer, and why the per-shot cost is what motivates a surrogate. |
| [03_surrogate.ipynb](03_surrogate.ipynb) | The learned surrogate: train it, benchmark it head to head against notebook 02, then find where it fails. Sensor dropout, ensemble error bars, and whether those error bars are calibrated. |

They are meant to be read in order. Each one ends by setting up the next.

## Running them

Notebooks 00 and 01 need the Open Fusion Toolkit installed, since they call
TokaMaker directly.

Notebooks 02 and 03 do **not**. They use the reduced forward model in
`src/ml/dataset.py`, which is pure NumPy, so they run anywhere the package
imports. That is also the honest limit of what they demonstrate, and both say
so in their opening cell.

From the repository root:

```bash
source .venv/bin/activate
jupyter lab notebooks/
```

Notebooks are committed **with their outputs**, matching the OFT examples, so
they can be read on GitHub without running anything.

## Relationship to the OFT examples

These are meant to sit beside the toolkit's own CUTE examples, not to replace
them. `CUTE_mesh_ex` builds the mesh, `CUTE_null_ex` assesses breakdown, and
`CUTE_pulse_ex` designs a full pulse. What none of them do, and what these add,
is **simulate a diagnostic**: computing what a sensor set would actually measure
for a given equilibrium. That is the step that makes noise studies, sensor-loss
studies, and learned surrogates possible.
