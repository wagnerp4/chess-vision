from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yolo(model_path: str | Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(model_path))
