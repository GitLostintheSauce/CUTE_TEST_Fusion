# Vacuum Vessel Eddy Current Compensation

## Background

CUTE's stainless steel vacuum vessel (resistivity η = 1.26×10⁻⁵ Ω·m) acts as a passive conductor. During transient events — current ramps, disruptions, or fast coil switching — time-varying magnetic flux induces eddy currents in the vessel wall. These eddy currents produce their own magnetic fields, which are measured by the flux loops and Mirnov probes alongside the plasma and coil signals.

Without compensation, eddy currents bias the equilibrium reconstruction, particularly during the ramp-up and ramp-down phases of a discharge.

## Physical Model

The vessel eddy current response is modeled as a sum of exponential decays:

```
H_vv(t) = Σ_k  A_k · exp(-t / τ_k)
```

where:
- **τ_k** are the wall eigenmode time constants (determined by the vessel geometry and resistivity)
- **A_k** is the amplitude matrix (n_sensors × n_modes) giving each sensor's response to mode k

For CUTE, the dominant modes have time constants in the range **30–105 μs**, consistent with a small, thin-walled vessel.

## Calibration Procedure

The vessel response is calibrated using a step-response simulation in TokaMaker:

1. **Baseline state**: Set all coil currents to zero (with 1μA baseline to avoid numerical issues), solve vacuum equilibrium.
2. **Step change**: Apply a step in one coil (default: CS01, 100A).
3. **Wall eigenmodes**: Compute wall eigenvalues via `eig_wall()`. These give the physical decay rates.
4. **TD simulation**: Run a time-domain simulation (`setup_td` / `step_td`) for 80 steps at dt=10μs.
5. **Eddy isolation**: Subtract the steady-state (vacuum) response from each TD snapshot. The residual is the eddy contribution.
6. **Amplitude fitting**: Given the fixed time constants from `eig_wall`, fit the amplitude matrix A via linear least-squares: `eddy(t) ≈ Σ_k A_k · exp(-t/τ_k)`.
7. **Normalization**: Divide amplitudes by the step magnitude to get per-unit response.

The result is a `VesselResponse` object containing `time_constants` and `amplitudes`.

**Important**: `eig_wall()` corrupts TokaMaker's vacuum solver state. All vacuum solves must complete before calling `eig_wall()`. The session-scoped fixture chain in `tests/conftest.py` enforces this ordering.

## Compensation Algorithm

During reconstruction of a time-series, the eddy contribution is subtracted using a recursive exponential filter:

```python
for each time step n:
    dI[n] = total_coil_current[n] - total_coil_current[n-1]
    for each mode k:
        α_k = exp(-dt / τ_k)
        state_k[n] = α_k * state_k[n-1] + dI[n]
        y_eddy[n] += A_k * state_k[n]

y_compensated[n] = y_measured[n] - y_eddy[n]
```

This O(n_times × n_modes) algorithm replaces the naive O(n_times²) convolution, making it practical for long time-series.

## Usage

### Computing the vessel response (one-time calibration)

```python
from src.reconstruct.eddy import compute_vessel_response, save_vessel_response

vr = compute_vessel_response(
    mygs, sensor_config,
    step_coil="CS01", step_amplitude=100.0,
    n_modes=3, dt=1e-5, n_steps=80,
)
save_vessel_response("data/vessel_response.h5", vr)
```

### Compensating measurements during reconstruction

```python
from src.reconstruct.eddy import load_vessel_response, compensate_eddy_fast

vr = load_vessel_response("data/vessel_response.h5")
compensated = compensate_eddy_fast(measurements, times, coil_currents, vr)
```

### Persistence

The `VesselResponse` is stored in HDF5 with datasets `time_constants` (n_modes,) and `amplitudes` (n_sensors, n_modes), plus a `coil_name` attribute.

## Validation

The eddy current model is validated by the following tests (see `tests/test_eddy.py`):

| Test | Criterion |
|------|-----------|
| Eddy non-zero | ≥10 sensors show eddy signal >1e-8 during transients |
| Time constants | All τ_k in [10μs, 100ms] (physical range) |
| Fit quality | R² > 0.90 for >70% of sensors with significant signal |
| HDF5 roundtrip | Save/load preserves all fields exactly |
| Compensation improves ramp | Compensated chi² < raw chi² for step response |
| Steady-state identity | Zero dI/dt → zero correction |
| Signal preservation | No NaN/Inf, RMS within 50% of original |

## Limitations

- The response is calibrated for a single coil (CS01). Multi-coil responses are approximated by summing total dI/dt across all coils.
- Only 3 eigenmodes are retained. Higher modes decay faster (<10μs) and are negligible for CUTE's diagnostic bandwidth.
- The model assumes linear superposition (valid for vacuum vessel response, not for saturated ferromagnetic structures).
