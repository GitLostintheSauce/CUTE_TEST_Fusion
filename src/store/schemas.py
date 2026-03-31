"""Pydantic schemas for CUTE pipeline data."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ShotMetadata(BaseModel):
    """Metadata for a single plasma shot."""

    shot_number: int
    timestamp: datetime
    coil_currents: dict[str, float] = Field(description="Coil currents in Ampere-turns")
    gas_pressure: float = Field(description="Gas pressure in Torr")
    operator_notes: str = ""


class SignalMetadata(BaseModel):
    """Metadata for a single diagnostic channel."""

    channel_id: str
    sensor_type: Literal["flux_loop", "mirnov"]
    position_r: float = Field(description="R position in meters")
    position_z: float = Field(description="Z position in meters")
    orientation: float = Field(description="Orientation angle in radians", default=0.0)


class EquilibriumResult(BaseModel):
    """Reconstructed equilibrium parameters for a single time slice."""

    plasma_current: float = Field(description="Plasma current in Amps")
    q95: float
    beta_poloidal: float
    internal_inductance: float
    boundary_r: list[float] = Field(description="Plasma boundary R coordinates in meters")
    boundary_z: list[float] = Field(description="Plasma boundary Z coordinates in meters")
