from __future__ import annotations

from pathlib import Path
from typing import Any, Type

import torch

RFDETR_SIZES = ("nano", "small", "base", "medium", "large")


def _import_rfdetr_classes() -> dict[str, Type[Any]]:
    try:
        from rfdetr import (
            RFDETRBase,
            RFDETRLarge,
            RFDETRMedium,
            RFDETRNano,
            RFDETRSmall,
        )
    except ImportError as exc:
        raise ImportError(
            "RF-DETR not installed. Please install: pip install rfdetr"
        ) from exc
    return {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "base": RFDETRBase,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }


def valid_rfdetr_sizes() -> tuple[str, ...]:
    return RFDETR_SIZES


def rfdetr_model_class(size: str) -> Type[Any]:
    key = size.lower().strip()
    model_map = _import_rfdetr_classes()
    if key not in model_map:
        raise ValueError(
            f"Invalid RF-DETR size: {size}. Must be one of: {list(model_map.keys())}"
        )
    return model_map[key]


def build_rfdetr(size: str, checkpoint: str | Path | None = None) -> Any:
    model_class = rfdetr_model_class(size)
    model = model_class()
    if checkpoint is not None and Path(checkpoint).exists():
        model.get_model().load_state_dict(
            torch.load(checkpoint, map_location="cpu")
        )
    return model
