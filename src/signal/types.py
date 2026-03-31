"""Data types for signal processing pipeline."""
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class RawSignal:
    """Raw sensor signal before processing."""

    timestamps: np.ndarray  # seconds
    values: np.ndarray  # raw ADC units
    channel_id: str
    sensor_type: Literal["flux_loop", "mirnov"]
    metadata: dict = field(default_factory=dict)


@dataclass
class ProcessedSignal:
    """Processed sensor signal with physical units."""

    timestamps: np.ndarray  # seconds
    values: np.ndarray  # physical units (Wb for flux loops, T for Mirnov)
    channel_id: str
    sensor_type: Literal["flux_loop", "mirnov"]
    processing_log: list[str] = field(default_factory=list)
