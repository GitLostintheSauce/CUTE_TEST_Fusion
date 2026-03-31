"""Phase 3c acceptance tests: Data schema and storage layer."""
from datetime import datetime, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from src.store import (
    EquilibriumResult,
    ShotMetadata,
    SignalData,
    SignalMetadata,
    index,
    load_shot,
    save_shot,
)


def _make_metadata(shot_number: int = 1) -> ShotMetadata:
    return ShotMetadata(
        shot_number=shot_number,
        timestamp=datetime(2026, 3, 31, tzinfo=timezone.utc),
        coil_currents={"CS01": 500.0, "PF01": 200.0},
        gas_pressure=1.5e-3,
        operator_notes="test shot",
    )


def _make_signal(channel_id: str = "FL01", sensor_type: str = "flux_loop") -> SignalData:
    meta = SignalMetadata(
        channel_id=channel_id,
        sensor_type=sensor_type,
        position_r=0.32,
        position_z=0.0,
        orientation=0.0,
    )
    return SignalData(
        metadata=meta,
        timestamps=np.linspace(0, 0.01, 100),
        values=np.sin(np.linspace(0, 2 * np.pi, 100)),
    )


def _make_equilibrium() -> EquilibriumResult:
    return EquilibriumResult(
        plasma_current=200e3,
        q95=3.5,
        beta_poloidal=30.0,
        internal_inductance=0.8,
        boundary_r=[0.15, 0.20, 0.35, 0.45, 0.35, 0.20, 0.15],
        boundary_z=[0.0, 0.25, 0.30, 0.0, -0.30, -0.25, 0.0],
    )


# --- Schema validation tests ---


def test_schema_valid_data():
    """[3c.1] Schema validates correct data."""
    meta = _make_metadata()
    assert meta.shot_number == 1

    sig = SignalMetadata(
        channel_id="FL01", sensor_type="flux_loop", position_r=0.3, position_z=0.0
    )
    assert sig.sensor_type == "flux_loop"

    eq = _make_equilibrium()
    assert eq.plasma_current == 200e3


def test_schema_rejects_invalid():
    """[3c.2] Schema rejects invalid data — at least 3 cases."""
    # Case 1: shot_number must be int
    with pytest.raises(ValidationError):
        ShotMetadata(
            shot_number="not_an_int",
            timestamp=datetime.now(tz=timezone.utc),
            coil_currents={},
            gas_pressure=1.0,
        )

    # Case 2: sensor_type must be "flux_loop" or "mirnov"
    with pytest.raises(ValidationError):
        SignalMetadata(
            channel_id="X",
            sensor_type="invalid_type",
            position_r=0.3,
            position_z=0.0,
        )

    # Case 3: missing required field
    with pytest.raises(ValidationError):
        EquilibriumResult(
            plasma_current=200e3,
            # missing q95, beta_poloidal, etc.
        )


# --- Round-trip tests ---


def test_roundtrip_metadata(tmp_path):
    """[3c.3] Save/load metadata round-trip."""
    meta = _make_metadata()
    path = tmp_path / "shot_001.h5"
    save_shot(path, meta)
    loaded = load_shot(path)
    assert loaded.metadata.shot_number == meta.shot_number
    assert loaded.metadata.gas_pressure == meta.gas_pressure
    assert loaded.metadata.coil_currents == meta.coil_currents
    assert loaded.metadata.operator_notes == meta.operator_notes


def test_roundtrip_signals(tmp_path):
    """[3c.4] Save/load signal arrays round-trip."""
    meta = _make_metadata()
    raw = {f"FL{i:02d}": _make_signal(f"FL{i:02d}") for i in range(1, 6)}
    path = tmp_path / "shot_001.h5"
    save_shot(path, meta, raw_signals=raw)
    loaded = load_shot(path)

    assert loaded.raw is not None
    assert len(loaded.raw) == 5
    for ch_id, sig in raw.items():
        assert ch_id in loaded.raw
        np.testing.assert_array_equal(loaded.raw[ch_id].values, sig.values)
        np.testing.assert_array_equal(loaded.raw[ch_id].timestamps, sig.timestamps)


def test_roundtrip_equilibrium(tmp_path):
    """[3c.5] Save/load equilibrium round-trip."""
    meta = _make_metadata()
    eq = [_make_equilibrium()]
    path = tmp_path / "shot_001.h5"
    save_shot(path, meta, equilibrium=eq)
    loaded = load_shot(path)

    assert loaded.equilibrium is not None
    assert len(loaded.equilibrium) == 1
    assert loaded.equilibrium[0].plasma_current == eq[0].plasma_current
    assert loaded.equilibrium[0].q95 == eq[0].q95
    assert loaded.equilibrium[0].boundary_r == eq[0].boundary_r


def test_hdf5_structure(tmp_path):
    """[3c.6] HDF5 file has correct group structure."""
    import h5py

    meta = _make_metadata()
    raw = {"FL01": _make_signal()}
    path = tmp_path / "shot_001.h5"
    save_shot(path, meta, raw_signals=raw)

    with h5py.File(path, "r") as f:
        assert "meta" in f
        assert "raw" in f
        assert "processed" in f
        assert "equilibrium" in f
        assert "FL01" in f["raw"]


def test_index(tmp_path):
    """[3c.7] Index returns correct summary for multiple shots."""
    for i in range(1, 4):
        meta = _make_metadata(shot_number=i)
        eq = [_make_equilibrium()]
        save_shot(tmp_path / f"shot_{i:03d}.h5", meta, equilibrium=eq)

    df = index(tmp_path)
    assert len(df) == 3
    assert "shot_number" in df.columns
    assert "timestamp" in df.columns
    assert "plasma_current" in df.columns
    assert set(df["shot_number"].tolist()) == {1, 2, 3}


def test_load_partial(tmp_path):
    """[3c.8] Load shot with only raw signals (no processed, no equilibrium)."""
    meta = _make_metadata()
    raw = {"FL01": _make_signal()}
    path = tmp_path / "shot_001.h5"
    save_shot(path, meta, raw_signals=raw)
    loaded = load_shot(path)

    assert loaded.raw is not None
    assert loaded.processed is None
    assert loaded.equilibrium is None
