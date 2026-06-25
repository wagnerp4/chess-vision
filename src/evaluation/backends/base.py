from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from src.evaluation.types import Detection, EvalConfig, ModelSpec


class DetectorBackend(Protocol):
    backend_id: str

    def load(self, spec: ModelSpec, root: Path | None = None) -> None: ...

    def predict(self, image_path: Path, eval_cfg: EvalConfig) -> list[Detection]: ...

    def predict_bgr(self, image_bgr: np.ndarray, eval_cfg: EvalConfig) -> list[Detection]: ...
