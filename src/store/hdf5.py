"""HDF5 storage backend for CUTE pipeline shot data."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .schemas import EquilibriumResult, ShotMetadata, SignalMetadata


@dataclass
class SignalData:
    """A single signal channel with metadata and time-series data."""

    metadata: SignalMetadata
    timestamps: np.ndarray
    values: np.ndarray


@dataclass
class Shot:
    """Complete shot data container."""

    metadata: ShotMetadata
    raw: dict[str, SignalData] | None = None
    processed: dict[str, SignalData] | None = None
    equilibrium: list[EquilibriumResult] | None = None


def save_shot(
    path: str | Path,
    metadata: ShotMetadata,
    raw_signals: dict[str, SignalData] | None = None,
    processed_signals: dict[str, SignalData] | None = None,
    equilibrium: list[EquilibriumResult] | None = None,
) -> None:
    """Save a shot to an HDF5 file.

    File structure:
        /meta           — shot metadata as JSON attributes
        /raw/{ch_id}    — raw signal datasets
        /processed/{ch_id} — processed signal datasets
        /equilibrium    — equilibrium results as JSON array
    """
    path = Path(path)
    with h5py.File(path, "w") as f:
        # Metadata
        meta_grp = f.create_group("meta")
        meta_grp.attrs["json"] = metadata.model_dump_json()

        # Raw signals
        raw_grp = f.create_group("raw")
        if raw_signals:
            for ch_id, sig in raw_signals.items():
                ch_grp = raw_grp.create_group(ch_id)
                ch_grp.create_dataset("timestamps", data=sig.timestamps)
                ch_grp.create_dataset("values", data=sig.values)
                ch_grp.attrs["metadata_json"] = sig.metadata.model_dump_json()

        # Processed signals
        proc_grp = f.create_group("processed")
        if processed_signals:
            for ch_id, sig in processed_signals.items():
                ch_grp = proc_grp.create_group(ch_id)
                ch_grp.create_dataset("timestamps", data=sig.timestamps)
                ch_grp.create_dataset("values", data=sig.values)
                ch_grp.attrs["metadata_json"] = sig.metadata.model_dump_json()

        # Equilibrium
        eq_grp = f.create_group("equilibrium")
        if equilibrium:
            eq_json = json.dumps([eq.model_dump() for eq in equilibrium])
            eq_grp.attrs["json"] = eq_json


def load_shot(path: str | Path) -> Shot:
    """Load a shot from an HDF5 file."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        # Metadata
        metadata = ShotMetadata.model_validate_json(f["meta"].attrs["json"])

        # Raw signals
        raw = None
        if len(f["raw"]) > 0:
            raw = {}
            for ch_id in f["raw"]:
                ch_grp = f["raw"][ch_id]
                sig_meta = SignalMetadata.model_validate_json(ch_grp.attrs["metadata_json"])
                raw[ch_id] = SignalData(
                    metadata=sig_meta,
                    timestamps=ch_grp["timestamps"][:],
                    values=ch_grp["values"][:],
                )

        # Processed signals
        processed = None
        if len(f["processed"]) > 0:
            processed = {}
            for ch_id in f["processed"]:
                ch_grp = f["processed"][ch_id]
                sig_meta = SignalMetadata.model_validate_json(ch_grp.attrs["metadata_json"])
                processed[ch_id] = SignalData(
                    metadata=sig_meta,
                    timestamps=ch_grp["timestamps"][:],
                    values=ch_grp["values"][:],
                )

        # Equilibrium
        equilibrium = None
        if "json" in f["equilibrium"].attrs:
            eq_list = json.loads(f["equilibrium"].attrs["json"])
            equilibrium = [EquilibriumResult.model_validate(eq) for eq in eq_list]

    return Shot(metadata=metadata, raw=raw, processed=processed, equilibrium=equilibrium)


def index(directory: str | Path) -> pd.DataFrame:
    """Scan a directory for shot HDF5 files and return a summary DataFrame."""
    directory = Path(directory)
    rows = []
    for h5_path in sorted(directory.glob("*.h5")):
        try:
            with h5py.File(h5_path, "r") as f:
                if "meta" not in f:
                    continue
                meta = ShotMetadata.model_validate_json(f["meta"].attrs["json"])
                row = {
                    "file": str(h5_path),
                    "shot_number": meta.shot_number,
                    "timestamp": meta.timestamp,
                    "gas_pressure": meta.gas_pressure,
                    "n_raw_channels": len(f["raw"]) if "raw" in f else 0,
                    "n_processed_channels": len(f["processed"]) if "processed" in f else 0,
                    "has_equilibrium": "json" in f.get("equilibrium", {}).attrs
                    if "equilibrium" in f
                    else False,
                }
                # Add plasma current if equilibrium exists
                if row["has_equilibrium"]:
                    eq_list = json.loads(f["equilibrium"].attrs["json"])
                    if eq_list:
                        row["plasma_current"] = eq_list[0]["plasma_current"]
                rows.append(row)
        except Exception:
            continue

    return pd.DataFrame(rows)
