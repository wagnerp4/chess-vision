from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.evaluation.types import Detection, EvalConfig, ModelSpec
from src.models.backends.ultralytics import load_yolo


class UltralyticsBackend:
    backend_id = "ultralytics"

    def __init__(self) -> None:
        self._model = None
        self._names: dict[int, str] = {}

    def load(self, spec: ModelSpec, root: Path | None = None) -> None:
        weights_path = Path(spec.weights)
        if root is not None and not weights_path.is_absolute():
            weights_path = root / weights_path
        if not weights_path.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        self._model = load_yolo(weights_path)
        raw_names = getattr(self._model, "names", {}) or {}
        if isinstance(raw_names, dict):
            self._names = {int(k): str(v) for k, v in raw_names.items()}
        else:
            self._names = {i: str(n) for i, n in enumerate(raw_names)}

    def set_class_names(self, id_to_name: dict[int, str]) -> None:
        self._names = id_to_name

    def predict(self, image_path: Path, eval_cfg: EvalConfig) -> list[Detection]:
        image_bgr = cv2.imread(str(image_path))
        return self.predict_bgr(image_bgr, eval_cfg)

    def predict_bgr(self, image_bgr: np.ndarray, eval_cfg: EvalConfig) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("Ultralytics backend not loaded")
        if image_bgr is None:
            return []
        results = self._model.predict(
            image_bgr,
            conf=eval_cfg.conf,
            iou=eval_cfg.iou_nms,
            imgsz=eval_cfg.imgsz,
            max_det=eval_cfg.max_det,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        out: list[Detection] = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            name = self._names.get(cls_id, f"class_{cls_id}")
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            out.append(
                Detection(
                    class_name=name,
                    bbox=[float(v) for v in xyxy],
                    confidence=float(boxes.conf[i]),
                )
            )
        return out
