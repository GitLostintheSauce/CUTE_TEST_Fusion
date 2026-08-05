# Notebooks

A followable walkthrough of the pipeline, in the style of the Open Fusion
Toolkit's `src/examples/TokaMaker` notebooks. This means that I have the physics motivation in
markdown, one idea per cell, and caveats stated where they apply rather than
collected in a footnote. I am still working on this.

CUTE is a teaching machine, so the pipeline is more useful as something a class
can reconstruct than as a finished application.

| Notebook | What it covers |
|---|---|
| [01_synthetic_diagnostics.ipynb](01_synthetic_diagnostics.ipynb) | The forward map: from a solved equilibrium to what each of the 130 magnetic sensors would read, with measurement noise and sensor loss. |

Planned: reconstruction from those signals, then the ML surrogate with its
benchmark and failure modes.

## Running them

These need the Open Fusion Toolkit installed, since they call TokaMaker
directly. From the repository root:

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
