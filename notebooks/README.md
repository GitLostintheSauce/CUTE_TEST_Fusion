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
| [01_synthetic_diagnostics.ipynb](01_synthetic_diagnostics.ipynb) | The forward map: from a solved equilibrium to what each of CUTE's 64 magnetic sensor channels would read, with measurement noise and sensor loss. Its stored outputs predate the switch to the real sensor layout; see below. |
| [02_reconstruction.ipynb](02_reconstruction.ipynb) | The inverse map: from sensor signals back to plasma parameters by least-squares inversion. Covers the inverse crime, what noise does to the answer (current and position survive it, minor radius does not), and why the per-shot cost is what motivates a surrogate. |
| [03_surrogate.ipynb](03_surrogate.ipynb) | The learned surrogate: train it, benchmark it head to head against notebook 02, then find where it fails. Sensor dropout, ensemble error bars, whether those error bars are calibrated, and why an R2 averaged over all four parameters hides the minor-radius failure. |
| [04_surrogate_pytorch.ipynb](04_surrogate_pytorch.ipynb) | The same surrogate in PyTorch. First proves it is the same model as notebook 03's (same predictions and same training steps from the same weights), then reruns the accuracy, speed, sensor-dropout and calibration studies side by side, and finds where aggregate numbers hide weaknesses: the minor radius, and relative error at low plasma current. |
| [05_what_the_sensors_can_see.ipynb](05_what_the_sensors_can_see.ipynb) | Where CUTE's real sensors are and what each measures, how the transcribed sensor table was checked, and why those sensors determine current and position but not minor radius: widening the current looks almost exactly like moving it outward. Uses the Cramér-Rao bound to show what *any* method could achieve, and what one sensor inside the plasma would change. |

They are meant to be read in order. Each one ends by setting up the next.
Notebook 05 can also be read straight after 02 or 03, whenever the minor-radius
result raises the question "why?".

Two older notebooks sit outside the sequence:

| Notebook | What it covers |
|---|---|
| [advanced_reconstruction.ipynb](advanced_reconstruction.ipynb) | EFIT-style fitting, eddy-current compensation and sensor-placement analysis, demonstrated on a made-up Green's matrix so it runs without OFT. It says clearly which numbers describe the methods and which would need the real solver. |
| [validation_report.ipynb](validation_report.ipynb) | How the noise and sensor-dropout tools behave, on placeholder measurements. Its summary table lists targets, not results computed in the notebook. |

## The sensor layout

The sensor positions are CUTE's real diagnostic layout, transcribed from the
CUTE group's diagnostics table into `config/cute_diagnostics.json`: 39 flux
loops and 25 magnetic probe channels, 64 in total. An earlier version of this
project used an invented layout of 130 sensors. Notebooks 01 to 05 and the two
older notebooks have all been rerun with the real one (01 with OFT
v26.9). Notebook 00 and the reference-equilibrium notebook do not use the sensors, so
the change does not affect them.

## Running them

Notebooks 00 and 01 need the Open Fusion Toolkit installed, since they call
TokaMaker directly.

Notebooks 02, 03 and 05 do **not**. They use the reduced forward model in
`src/ml/dataset.py`, which is pure NumPy, so they run anywhere the package
imports. That is also the honest limit of what they demonstrate, and each
says so in its opening cell. The two older notebooks do not need OFT either.

Notebook 04 has the same scope as 03 and additionally needs PyTorch
(`pip install -e ".[dev,torch]"`). It runs on the CPU; no GPU is needed.
It takes about three minutes, most of it retraining with several seeds.

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
