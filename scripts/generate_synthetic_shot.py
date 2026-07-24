"""Generate synthetic shot HDF5 files with realistic data for dashboard viewing.

Produces several distinct shots (different peak current, shape, and duration)
so the shot browser behaves like a real shot-review tool. Shot 1 keeps its
original seed and parameters for reproducibility.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime

import numpy as np

from src.forward.sensors import generate_cute_sensors
from src.store.hdf5 import SignalData, save_shot
from src.store.schemas import EquilibriumResult, ShotMetadata, SignalMetadata

# Each entry: (shot_number, seed, ip_max [A], duration [s], kappa, delta,
#              gas_pressure [Torr], timestamp, operator_notes)
SHOT_SPECS = [
    (1, 42, 200e3, 0.010, 1.7, 0.40, 1.5e-3,
     datetime(2026, 4, 1, 12, 0, 0),
     "Synthetic shot for dashboard demo"),
    (2, 7, 150e3, 0.008, 1.5, 0.30, 1.2e-3,
     datetime(2026, 4, 1, 12, 15, 0),
     "Synthetic: lower current, reduced elongation"),
    (3, 19, 240e3, 0.012, 1.9, 0.45, 1.8e-3,
     datetime(2026, 4, 1, 12, 30, 0),
     "Synthetic: high current, strongly shaped"),
]


def generate_shot(shot_number, seed, ip_max, duration, kappa, delta,
                  gas_pressure, timestamp, notes, out_dir):
    rng = np.random.default_rng(seed)
    sensor_config = generate_cute_sensors()

    # Time base: 100kHz sampling
    fs = 100_000
    n_samples = int(fs * duration)
    timestamps = np.linspace(0, duration, n_samples, endpoint=False)

    # Plasma current ramp-up profile
    ip_profile = ip_max * (1 - np.exp(-timestamps / 0.002))

    meta = ShotMetadata(
        shot_number=shot_number,
        timestamp=timestamp,
        coil_currents={f"CS{i:02d}": 500.0 for i in range(1, 15)},
        gas_pressure=gas_pressure,
        operator_notes=notes,
    )

    raw_signals = {}
    processed_signals = {}

    # Flux loop signals: psi varies with Ip ramp
    for sensor in sensor_config.flux_loops:
        base_psi = 0.01 * (1 + 0.5 * sensor["R"]) * ip_profile / 200e3
        noise = rng.normal(0, 0.001, n_samples)
        pickup = 0.0005 * np.sin(2 * np.pi * 60 * timestamps)
        raw_values = base_psi + noise + pickup

        sig_meta = SignalMetadata(
            channel_id=sensor["id"],
            sensor_type="flux_loop",
            position_r=sensor["R"],
            position_z=sensor["Z"],
        )
        raw_signals[sensor["id"]] = SignalData(
            metadata=sig_meta, timestamps=timestamps, values=raw_values,
        )
        processed_signals[sensor["id"]] = SignalData(
            metadata=sig_meta, timestamps=timestamps, values=base_psi,
        )

    # Mirnov probe signals: B varies with Ip ramp
    for sensor in sensor_config.mirnov_probes:
        base_B = 0.1 * (1 + 0.3 * sensor["R"]) * ip_profile / 200e3
        noise = rng.normal(0, 0.005, n_samples)
        pickup = 0.002 * np.sin(2 * np.pi * 60 * timestamps)
        raw_values = base_B + noise + pickup

        sig_meta = SignalMetadata(
            channel_id=sensor["id"],
            sensor_type="mirnov",
            position_r=sensor["R"],
            position_z=sensor["Z"],
            orientation=sensor["angle"],
        )
        raw_signals[sensor["id"]] = SignalData(
            metadata=sig_meta, timestamps=timestamps, values=raw_values,
        )
        processed_signals[sensor["id"]] = SignalData(
            metadata=sig_meta, timestamps=timestamps, values=base_B,
        )

    # Equilibrium results at 5 evenly spaced time slices
    equilibrium = []
    time_slices = np.linspace(0.1 * duration, 0.9 * duration, 5)
    for t in time_slices:
        ip_at_t = float(ip_max * (1 - np.exp(-t / 0.002)))
        frac = ip_at_t / ip_max

        # Boundary shape scales with plasma current
        n_bdy = 60
        theta = np.linspace(0, 2 * np.pi, n_bdy, endpoint=False)
        a = 0.15 * frac + 0.02
        bdy_r = (0.32 + a * np.cos(theta + delta * np.sin(theta))).tolist()
        bdy_z = (kappa * a * np.sin(theta)).tolist()

        equilibrium.append(EquilibriumResult(
            plasma_current=ip_at_t,
            q95=3.9 + 0.5 * (1 - frac),
            beta_poloidal=30.0 * frac,
            internal_inductance=1.3 + 0.1 * (1 - frac),
            boundary_r=bdy_r,
            boundary_z=bdy_z,
        ))

    out_path = os.path.join(out_dir, f"shot_{shot_number:03d}.h5")
    save_shot(out_path, meta,
              raw_signals=raw_signals,
              processed_signals=processed_signals,
              equilibrium=equilibrium)

    print(f"Synthetic shot saved to {out_path}")
    print(f"  {len(raw_signals)} raw channels, {len(processed_signals)} processed channels")
    print(f"  {len(equilibrium)} equilibrium time slices")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
    os.makedirs(out_dir, exist_ok=True)
    for spec in SHOT_SPECS:
        generate_shot(*spec, out_dir=out_dir)


if __name__ == "__main__":
    main()
