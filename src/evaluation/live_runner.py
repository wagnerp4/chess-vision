from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.inference.realtime import (
    DEFAULT_PIECE_EVERY,
    LiveConfig,
    default_model,
    live_loop,
)


def default_piece_every(backend: str) -> int:
    return DEFAULT_PIECE_EVERY.get(backend.lower(), 3)


def resolve_live_weights(
    backend: str,
    weights: str | None,
    root: Path | None = None,
) -> str:
    if weights:
        path = Path(weights)
        if root is not None and not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"Weights not found: {path}")
        return str(path.resolve())
    if backend == "ultralytics":
        model_path = default_model()
        return str(model_path.resolve())
    raise ValueError(f"--weights is required for backend '{backend}'")


def run_live_benchmark(
    config: LiveConfig,
    root: Path | None = None,
) -> dict[str, Any]:
    latency = live_loop(config, root=root)
    elapsed = (
        sum(latency.frame_interval_ms) / 1000.0 if latency.frame_interval_ms else 0.0
    )
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": "oak_live",
        "model": {
            "backend": config.backend,
            "weights": str(Path(config.weights).resolve()),
            "label": config.model_label,
            "rfdetr_size": config.rfdetr_size,
        },
        "live_config": {
            "piece_every": config.piece_every,
            "grid_every": config.grid_every,
            "redetect_every": config.redetect_every,
            "camera_size": list(config.camera_size),
            "duration_sec": config.duration_sec,
            "eval": config.eval.__dict__,
        },
        "elapsed_seconds": round(elapsed, 2),
        "latency": latency.summary(config.piece_every),
        "status": "ok",
    }
    return payload


def write_live_benchmark(payload: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path
