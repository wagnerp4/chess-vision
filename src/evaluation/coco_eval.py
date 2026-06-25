from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.evaluation.dataset import detections_to_legacy_dict
from src.evaluation.types import Detection, SplitData
from src.metrics import evaluate_model_predictions
from src.metrics.tasks.obj_det.obj_det_metrics import calculate_iou


def _build_coco_dataset(
    split_data: SplitData,
    predictions: dict[str, list[Detection]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    categories = [
        {"id": cid, "name": name, "supercategory": "piece"}
        for cid, name in sorted(split_data.id_to_name.items())
    ]
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1

    for img_idx, (stem, (w, h)) in enumerate(split_data.image_sizes.items(), start=1):
        images.append({"id": img_idx, "file_name": stem, "width": w, "height": h})
        for gt in split_data.ground_truth.get(stem, []):
            if gt.class_name not in split_data.name_to_id:
                continue
            x1, y1, x2, y2 = gt.bbox
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": split_data.name_to_id[gt.class_name],
                    "bbox": [x1, y1, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    stem_to_img_id = {stem: idx for idx, stem in enumerate(split_data.image_sizes.keys(), start=1)}
    coco_results: list[dict[str, Any]] = []
    for stem, preds in predictions.items():
        if stem not in stem_to_img_id:
            continue
        image_id = stem_to_img_id[stem]
        for pred in preds:
            if pred.class_name not in split_data.name_to_id:
                continue
            x1, y1, x2, y2 = pred.bbox
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id": split_data.name_to_id[pred.class_name],
                    "bbox": [x1, y1, bw, bh],
                    "score": pred.confidence,
                }
            )

    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return dataset, coco_results


def evaluate_coco_metrics(
    split_data: SplitData,
    predictions: dict[str, list[Detection]],
) -> dict[str, Any]:
    if not split_data.image_sizes:
        return {
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "per_class_ap50": {},
        }

    if not any(split_data.ground_truth.values()):
        return {
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "per_class_ap50": {},
        }

    dataset, coco_results = _build_coco_dataset(split_data, predictions)
    if not dataset["annotations"]:
        return {
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "per_class_ap50": {},
        }

    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()

    if not coco_results:
        precision, recall = _global_precision_recall(
            split_data,
            predictions,
            iou_threshold=0.5,
        )
        return {
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "precision": precision,
            "recall": recall,
            "per_class_ap50": {},
        }

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    per_class_ap50: dict[str, float] = {}
    if coco_eval.eval is not None:
        precision = coco_eval.eval["precision"]
        cat_ids = coco_eval.params.catIds
        for idx, cat_id in enumerate(cat_ids):
            ap_slice = precision[0, :, idx, 0, -1]
            ap_slice = ap_slice[ap_slice > -1]
            if len(ap_slice) == 0:
                ap50 = 0.0
            else:
                ap50 = float(np.mean(ap_slice))
            name = split_data.id_to_name.get(cat_id, f"class_{cat_id}")
            per_class_ap50[name] = ap50

    precision, recall = _global_precision_recall(
        split_data,
        predictions,
        iou_threshold=0.5,
    )
    return {
        "mAP50": float(stats[1]) if len(stats) > 1 else 0.0,
        "mAP50_95": float(stats[0]) if len(stats) > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "per_class_ap50": per_class_ap50,
    }


def _global_precision_recall(
    split_data: SplitData,
    predictions: dict[str, list[Detection]],
    iou_threshold: float,
) -> tuple[float, float]:
    tp = 0
    fp = 0
    fn = 0
    for stem, gt_items in split_data.ground_truth.items():
        preds = sorted(
            predictions.get(stem, []),
            key=lambda d: d.confidence,
            reverse=True,
        )
        matched = [False] * len(gt_items)
        for pred in preds:
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gt_items):
                if matched[idx]:
                    continue
                if pred.class_name != gt.class_name:
                    continue
                iou = calculate_iou(pred.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                tp += 1
                matched[best_idx] = True
            else:
                fp += 1
        fn += sum(1 for m in matched if not m)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def evaluate_legacy_metrics(
    split_data: SplitData,
    predictions: dict[str, list[Detection]],
    iou_threshold: float,
) -> dict[str, Any]:
    gt_legacy = {
        stem: detections_to_legacy_dict(dets)
        for stem, dets in split_data.ground_truth.items()
    }
    pred_legacy = {
        stem: detections_to_legacy_dict(dets)
        for stem, dets in predictions.items()
    }
    metrics = evaluate_model_predictions(
        pred_legacy,
        gt_legacy,
        iou_threshold=iou_threshold,
    )
    return {
        "mAP50": float(metrics["mAP"]),
        "note": "11-point per-image mean AP",
        "num_images": int(metrics["num_images"]),
    }


def evaluate_dual_metrics(
    split_data: SplitData,
    predictions: dict[str, list[Detection]],
    iou_match: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, float]]]:
    coco = evaluate_coco_metrics(split_data, predictions)
    legacy = evaluate_legacy_metrics(split_data, predictions, iou_match)
    per_class: dict[str, dict[str, float]] = {}
    for name, ap50 in coco.get("per_class_ap50", {}).items():
        per_class[name] = {"coco_mAP50": ap50}
    return coco, legacy, per_class
