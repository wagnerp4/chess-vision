from __future__ import annotations

from pathlib import Path

from src.evaluation.backends.base import DetectorBackend
from src.evaluation.backends.rfdetr import RfdetrBackend
from src.evaluation.backends.ultralytics import UltralyticsBackend
from src.evaluation.types import ModelSpec

_BACKENDS: dict[str, type] = {
    "ultralytics": UltralyticsBackend,
    "rfdetr": RfdetrBackend,
}


def get_backend(spec: ModelSpec) -> DetectorBackend:
    backend_key = spec.backend.lower().strip()
    if backend_key not in _BACKENDS:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"Unknown backend '{spec.backend}'. Supported: {supported}"
        )
    return _BACKENDS[backend_key]()
