from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


class BaseObjDetMetric(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def update(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> None:
        pass

    @abstractmethod
    def compute(self) -> float:
        pass


class IoUMetric(BaseObjDetMetric):
    def __init__(self) -> None:
        self._values: list[float] = []

    def reset(self) -> None:
        self._values = []

    def update(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> None:
        for pred in predictions:
            pred_bbox = pred.get("bbox", [])
            best = 0.0
            for gt in ground_truths:
                if pred.get("class") == gt.get("class"):
                    best = max(best, calculate_iou(pred_bbox, gt.get("bbox", [])))
            self._values.append(best)

    def compute(self) -> float:
        if not self._values:
            return 0.0
        return float(np.mean(self._values))


class AveragePrecisionMetric(BaseObjDetMetric):
    def __init__(self, iou_threshold: float = 0.5) -> None:
        self._iou_threshold = iou_threshold
        self._predictions: list[dict[str, Any]] = []
        self._ground_truths: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._predictions = []
        self._ground_truths = []

    def update(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> None:
        self._predictions = list(predictions)
        self._ground_truths = list(ground_truths)

    def compute(self) -> float:
        return _ap_from_lists(
            self._predictions,
            self._ground_truths,
            self._iou_threshold,
        )


class MeanAveragePrecisionMetric(BaseObjDetMetric):
    def __init__(self, iou_threshold: float = 0.5) -> None:
        self._iou_threshold = iou_threshold
        self._aps: list[float] = []

    def reset(self) -> None:
        self._aps = []

    def update(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> None:
        self._aps.append(
            _ap_from_lists(predictions, ground_truths, self._iou_threshold)
        )

    def compute(self) -> float:
        if not self._aps:
            return 0.0
        return float(np.mean(self._aps))


def _ap_from_lists(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float,
) -> float:
    if not predictions or not ground_truths:
        return 0.0

    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))

    predictions_sorted = sorted(
        predictions,
        key=lambda x: x.get("confidence", 0.0),
        reverse=True,
    )
    gt_matched = [False] * len(ground_truths)

    for i, pred in enumerate(predictions_sorted):
        best_iou = 0.0
        best_gt_idx = -1
        pred_class = pred.get("class", "")
        pred_bbox = pred.get("bbox", [])

        for j, gt in enumerate(ground_truths):
            if gt_matched[j]:
                continue
            if pred_class == gt.get("class", ""):
                iou = calculate_iou(pred_bbox, gt.get("bbox", []))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / max(len(ground_truths), 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

    ap = 0.0
    for r in np.arange(0, 1.1, 0.1):
        precisions_at_recall = precisions[recalls >= r]
        if len(precisions_at_recall) > 0:
            ap += float(np.max(precisions_at_recall)) / 11.0
    return ap


def evaluate_model_predictions(
    predictions: dict[str, list[dict[str, Any]]],
    ground_truths: dict[str, list[dict[str, Any]]],
    iou_threshold: float = 0.5,
) -> dict[str, float | list[float] | int]:
    aps: list[float] = []
    for image_id in ground_truths:
        if image_id in predictions:
            aps.append(
                _ap_from_lists(
                    predictions[image_id],
                    ground_truths[image_id],
                    iou_threshold,
                )
            )
    return {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "AP_per_image": aps,
        "num_images": len(aps),
    }
