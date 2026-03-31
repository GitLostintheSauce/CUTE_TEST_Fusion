"""CUTE pipeline signal processing library."""
from .processing import ProcessingConfig, process_shot, process_signal
from .types import ProcessedSignal, RawSignal

__all__ = [
    "ProcessingConfig",
    "ProcessedSignal",
    "RawSignal",
    "process_shot",
    "process_signal",
]
