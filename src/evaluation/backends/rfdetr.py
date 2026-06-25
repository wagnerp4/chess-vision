from __future__ import annotations

from pathlib import Path

import torch
import numpy as np

from src.evaluation.backends.base import DetectorBackend
from src.evaluation.types import Detection, EvalConfig, ModelSpec
from src.models.backends.rfdetr import build_rfdetr, rfdetr_model_class

RFDETR_SCAFFOLD_NAMES = (
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "checkpoint_best_total.pth",
)


def _load_lightning_ckpt(model_cls, ckpt_path: Path):
    scaffold_path: Path | None = None
    for name in RFDETR_SCAFFOLD_NAMES:
        candidate = ckpt_path.parent / name
        if candidate.is_file():
            scaffold_path = candidate
            break
    if scaffold_path is None:
        raise FileNotFoundError(
            f"Lightning checkpoint {ckpt_path} requires a sibling .pth scaffold "
            f"({', '.join(RFDETR_SCAFFOLD_NAMES)}) with training args."
        )
    model = model_cls.from_checkpoint(str(scaffold_path))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", {})
    stripped = {
        key.replace("model.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not stripped:
        raise ValueError(f"No model.* tensors found in Lightning checkpoint: {ckpt_path}")
    target = model.model.model
    target.load_state_dict(stripped, strict=False)
    return model


def predictions_from_supervision(det, id_to_name: dict[int, str]) -> list[Detection]:
    if det is None or len(det) == 0:
        return []
    xyxy = np.asarray(det.xyxy)
    class_ids = np.asarray(det.class_id) if det.class_id is not None else np.zeros(len(det))
    confidences = (
        np.asarray(det.confidence)
        if det.confidence is not None
        else np.ones(len(det))
    )
    out: list[Detection] = []
    for i in range(len(det)):
        cid = int(class_ids[i])
        name = id_to_name.get(cid, f"class_{cid}")
        out.append(
            Detection(
                class_name=name,
                bbox=xyxy[i].tolist(),
                confidence=float(confidences[i]),
            )
        )
    return out


class RfdetrBackend:
    backend_id = "rfdetr"

    def __init__(self) -> None:
        self._model = None
        self._id_to_name: dict[int, str] = {}

    def load(self, spec: ModelSpec, root: Path | None = None) -> None:
        weights_path = Path(spec.weights)
        if root is not None and not weights_path.is_absolute():
            weights_path = root / weights_path
        if not weights_path.is_file():
            raise FileNotFoundError(f"RF-DETR checkpoint not found: {weights_path}")

        size = spec.size or "large"
        suffix = weights_path.suffix.lower()
        model_cls = rfdetr_model_class(size)
        if suffix == ".ckpt":
            self._model = _load_lightning_ckpt(model_cls, weights_path)
        elif suffix in (".pth", ".pt"):
            payload = torch.load(str(weights_path), map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "args" in payload:
                self._model = model_cls.from_checkpoint(str(weights_path))
            else:
                self._model = build_rfdetr(size, checkpoint=weights_path)
        else:
            raise ValueError(
                f"Unsupported RF-DETR checkpoint format: {weights_path.suffix}. "
                "Use .ckpt, .pth, or portable .pt"
            )

    def set_class_names(self, id_to_name: dict[int, str]) -> None:
        self._id_to_name = id_to_name

    def predict(self, image_path: Path, eval_cfg: EvalConfig) -> list[Detection]:
        import cv2

        image_bgr = cv2.imread(str(image_path))
        return self.predict_bgr(image_bgr, eval_cfg)

    def predict_bgr(self, image_bgr: np.ndarray, eval_cfg: EvalConfig) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("RF-DETR backend not loaded")
        if image_bgr is None:
            return []
        import cv2

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        det = self._model.predict(image_rgb, threshold=eval_cfg.conf)
        return predictions_from_supervision(det, self._id_to_name)
