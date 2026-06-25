from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.board_stream import BOARD_PX
from src.evaluation.types import EvalConfig
from src.inference.realtime import (
    DEFAULT_GRID_EVERY,
    DEFAULT_REDETECT_EVERY,
    LiveConfig,
    LiveLatencyTracker,
    default_model,
    draw_detections,
    live_loop as _live_loop,
    run_yolo,
)

FrameCallback = Callable[[dict[str, Any], np.ndarray | None, np.ndarray | None], None]

DETECT_EVERY = DEFAULT_GRID_EVERY
REDETECT_EVERY = DEFAULT_REDETECT_EVERY

__all__ = [
    "BOARD_PX",
    "DETECT_EVERY",
    "REDETECT_EVERY",
    "FrameCallback",
    "LiveConfig",
    "LiveLatencyTracker",
    "default_model",
    "draw_detections",
    "live_loop",
    "run_yolo",
]


def live_loop(
    model_path: str,
    conf: float = 0.25,
    yolo_every: int = 3,
    show_windows: bool = True,
    on_frame: FrameCallback | None = None,
    snap_dir: Path | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    config = LiveConfig(
        backend="ultralytics",
        weights=model_path,
        model_label=Path(model_path).name,
        eval=EvalConfig(conf=conf),
        piece_every=yolo_every,
        show_windows=show_windows,
        snap_dir=snap_dir,
    )
    _live_loop(config, on_frame=on_frame, stop_event=stop_event)
