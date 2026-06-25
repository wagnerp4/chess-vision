from src.evaluation.backends.base import DetectorBackend
from src.evaluation.backends.registry import get_backend
from src.evaluation.backends.rfdetr import RfdetrBackend
from src.evaluation.backends.ultralytics import UltralyticsBackend

__all__ = [
    "DetectorBackend",
    "RfdetrBackend",
    "UltralyticsBackend",
    "get_backend",
]
