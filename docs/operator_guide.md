# Operator Guide

## Processing a new shot

### 1. Save raw data

Place the raw shot HDF5 file in `data/raw/`. The file should follow the schema defined in `src/store/schemas.py`:

```python
from src.store.hdf5 import save_shot, SignalData
from src.store.schemas import ShotMetadata, SignalMetadata

meta = ShotMetadata(
    shot_number=42,
    timestamp=datetime.now(),
    coil_currents={"CS01": 500.0, "CS02": 500.0, ...},
    gas_pressure=1.0e-3,
)
save_shot("data/raw/shot_042.h5", meta, raw_signals=signals)
```

### 2. Process signals

```python
from src.signal.processing import ProcessingConfig, process_pipeline

config = ProcessingConfig.from_toml("config/processing.toml")
processed = process_pipeline(raw_signals, config)
```

### 3. Run reconstruction

```bash
cute-reconstruct --shot data/raw/shot_042.h5 --output data/processed/shot_042.h5
```

Or programmatically:

```python
from src.reconstruct import fit_equilibrium
from src.forward.sensors import generate_cute_sensors

sensor_config = generate_cute_sensors()
result = fit_equilibrium(mygs, measurements, sensor_config)
```

### 4. View results

```bash
python -m src.dashboard.app --port 8050
```

Open `http://localhost:8050` in your browser. Click "Refresh" to scan for shots, then click a row to load it.

## Using the dashboard

### Shot Browser
- Click **Refresh** to scan the `data/` directory for shot HDF5 files
- Click a row to load that shot's data
- The table shows shot number, timestamp, channel counts, and whether equilibrium data exists

### Signal Viewer
- Select a channel from the dropdown to plot raw vs. processed signals
- Both traces are overlaid with the raw signal semi-transparent

### Equilibrium Viewer
- Shows the reconstructed flux surfaces, plasma boundary, vessel wall, and sensor positions
- Use the time slider to step through time slices
- Red thick line = plasma boundary, gray dashed = vessel wall
- Green dots = flux loops, orange diamonds = Mirnov probes

### Parameter Timeline
- Plots Ip, q95, beta_poloidal, and internal inductance vs. time index
- Ip on left y-axis, dimensionless quantities on right y-axis

### Sim vs. Experiment
- Select a channel to compare measured signal with synthetic (forward model) prediction
- Three traces: Measured, Synthetic, Residual

### Remote Access
Set the `CUTE_DASH_TOKEN` environment variable or pass `--token` to enable token-based auth:
```bash
python -m src.dashboard.app --token mysecrettoken --port 8050
```

## EFIT Reconstruction (Green's function method)

The EFIT method provides an alternative to constraint-based reconstruction by explicitly decomposing measurements into coil and plasma contributions.

### Pre-compute the Green's function matrix

```python
from src.forward.sensors import generate_cute_sensors
from src.reconstruct.greens import compute_greens_matrix, save_greens_matrix

sensor_config = generate_cute_sensors()
G, coil_names = compute_greens_matrix(mygs, sensor_config)
save_greens_matrix("data/greens_matrix.h5", G, coil_names)
```

### Run EFIT reconstruction

```python
from src.reconstruct.greens import load_greens_matrix
from src.reconstruct.efit import efit_reconstruct

G, coil_names = load_greens_matrix("data/greens_matrix.h5")
result = efit_reconstruct(mygs, measurements, sensor_config, G, coil_names)
print(f"Ip = {result.equilibrium.plasma_current:.0f} A")
print(f"Chi² = {result.diagnostics.chi_squared:.2e}")
```

## Eddy Current Compensation

During current ramps, vacuum vessel eddy currents bias the measurements. See [docs/eddy_currents.md](eddy_currents.md) for the full theory.

### One-time calibration

```python
from src.reconstruct.eddy import compute_vessel_response, save_vessel_response

vr = compute_vessel_response(mygs, sensor_config, step_coil="CS01", step_amplitude=100.0)
save_vessel_response("data/vessel_response.h5", vr)
```

### Apply compensation before reconstruction

```python
from src.reconstruct.eddy import load_vessel_response, compensate_eddy_fast

vr = load_vessel_response("data/vessel_response.h5")
compensated = compensate_eddy_fast(measurements, times, coil_currents, vr)
```

## Sensor Placement Analysis

Evaluate which sensors contribute most to reconstruction accuracy:

```python
from src.validation.sensor_placement import leave_one_out, find_minimum_viable_set
from src.reconstruct.constraints import sensor_ids_ordered

sensor_ids = sensor_ids_ordered(sensor_config)

# Which sensor matters most?
degradation = leave_one_out(G, sensor_ids)
for sid, deg in sorted(degradation.items(), key=lambda x: -x[1])[:10]:
    print(f"{sid}: degradation = {deg:.2e}")

# What's the minimum viable set?
viable_ids, n_viable = find_minimum_viable_set(G, sensor_ids, coil_names)
print(f"Minimum viable set: {n_viable} sensors")
```

## Changing the sensors

The sensor positions live in one data file, `config/cute_diagnostics.json`,
transcribed from CUTE's diagnostics table. `src/forward/sensors.py` reads it;
nothing else in the code hard-codes a sensor position or count.

### Moving or adding a flux loop or probe

1. Edit `config/cute_diagnostics.json`. Flux loops need a `name`, `R` and `Z`
   in meters; magnetic probes also need `phi_deg` (toroidal angle) and the
   unit vector `N_R`, `N_Z` of the direction they measure.
2. Run `pytest tests/test_forward.py`. It checks that every sensor sits on the
   vessel, which catches typos and centimeter-for-meter slips.
3. Everything computed from the sensors is now stale. Regenerate it:
   `python scripts/generate_synthetic_shot.py`,
   `python scripts/train_surrogate.py --samples 8000 --epochs 400`,
   `python scripts/validate_surrogate.py`, `python scripts/uncertainty_report.py`,
   and `python scripts/plot_sensor_layout.py`, then rerun the notebooks.

Probe entries that repeat another probe's position and direction are dropped
automatically, because a 2D axisymmetric model would read them identically.
See Part 9 of `docs/primer.md` for why.

### Adding a new sensor type

#### 1. Define the sensor geometry

Add a new list to `config/cute_diagnostics.json` (for example
`"rogowski_coils"`), and load it in `generate_cute_sensors()` in
`src/forward/sensors.py`, alongside the flux loops and probes.

#### 2. Add the forward model

Edit `src/forward/model.py` and add a new evaluation function:

```python
def rogowski_coil(eval_func, sensor_pos, ...):
    """Evaluate the Rogowski coil signal."""
    ...
```

#### 3. Update the signal metadata schema

Add the new sensor type to `src/store/schemas.py`:

```python
class SignalMetadata(BaseModel):
    sensor_type: Literal["flux_loop", "mirnov", "rogowski"]
```

#### 4. Add tests

Create tests in `tests/test_forward.py` following the existing pattern.

## How to troubleshoot common issues

### "Only one TokaMaker instance per kernel"

OFT enforces a single TokaMaker instance per Python process. If you see errors about duplicate instances:
- Restart your Python kernel/process
- Use the session-scoped fixture in tests (`tokamaker_session`)
- For parallel processing, use `multiprocessing` (separate processes)

### TokaMaker solve fails or produces bad results

- Check that `init_psi()` was called before the first `solve()`
- Verify coil bounds are set (`set_coil_bounds`)
- Check that profiles are set (`set_profiles`)
- Try reducing the Ip target if the solver diverges

### Signal processing filters are unstable

- Ensure you're using SOS format (default in our pipeline)
- Check that the sample rate is set correctly
- Bandpass filter limits must satisfy: `low > 0` and `high < fs/2`

### q95 returns 0 or is unavailable

TokaMaker's flux surface tracing sometimes fails for strongly shaped or diverted equilibria. This is a known upstream limitation. The pipeline handles this by returning 0.0 for q95 when tracing fails.

### Dashboard doesn't find any shots

- Click "Refresh" to re-scan the data directory
- Ensure shot files are in `data/raw/`, `data/processed/`, or `data/synthetic/`
- Files must be valid HDF5 with the expected schema (see `src/store/hdf5.py`)

### HDF5 file is corrupted

- Always close files properly (use context managers)
- Don't write to a file while another process is reading it
- Keep backups of raw data — processed data can be regenerated
